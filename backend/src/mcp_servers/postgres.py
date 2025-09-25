import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

import asyncpg
from fastmcp import Context, FastMCP
from pydantic import BaseModel

from mcp_servers.utils.constants import MCP_HOST, MCP_PORT
from mcp_servers.utils.helper import log, start_mcp_server
from mcp_servers.utils.models import MCPResponse

logger = logging.getLogger(__name__)

# Database configuration
DB_CONNECTION_STRING = os.getenv("DB_CONNECTION_STRING", "postgresql://app:app@localhost:5432/app")


@dataclass
class DatabaseObject:
    name: str
    type: str
    schema: str


@dataclass
class ObjectDetails:
    name: str
    type: str
    schema: str
    columns: List[Dict[str, Any]]
    constraints: List[Dict[str, Any]]
    comments: Optional[str] = None


class TableInfo(BaseModel):
    schema: str
    name: str
    object_type: str  # 'table', 'view', 'materialized view'
    object_comment: Optional[str] = None
    definition: Optional[str] = None  # For views and materialized views
    is_updatable: Optional[bool] = None  # For views


class OtherObjectInfo(BaseModel):
    name: str
    type: str  # 'function', 'procedure', 'sequence', 'type', 'domain', 'enum', etc.
    comment: Optional[str] = None


class SchemaDetails(BaseModel):
    schema: str
    tables: List[TableInfo]
    other_objects: List[OtherObjectInfo]


class PostgreSQLClient:
    def __init__(self):
        self.connection_pool: Optional[asyncpg.Pool] = None

    async def connect(self):
        """Initialize database connection pool"""
        try:
            self.connection_pool = await asyncpg.create_pool(
                DB_CONNECTION_STRING,
                min_size=1,
                max_size=10
            )
            logger.info("Connected to PostgreSQL database using connection string")
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            raise

    async def disconnect(self):
        """Close database connection pool"""
        if self.connection_pool:
            await self.connection_pool.close()
            logger.info("Disconnected from PostgreSQL database")

    async def execute_query(self, query: str, params: tuple = ()) -> List[Dict[str, Any]]:
        """Execute a SQL query and return results"""
        if not self.connection_pool:
            await self.connect()
        
        async with self.connection_pool.acquire() as conn:
            try:
                rows = await conn.fetch(query, *params)
                return [dict(row) for row in rows]
            except Exception as e:
                logger.error(f"Error executing query: {e}")
                raise

    async def list_schemas(self) -> List[Dict[str, Any]]:
        """List all user-defined schemas in the database with their comments"""
        query = """
        SELECT 
            s.schema_name,
            obj_description(n.oid, 'pg_namespace') AS comment
        FROM information_schema.schemata s
        LEFT JOIN pg_namespace n ON n.nspname = s.schema_name
        WHERE s.schema_name NOT IN (
            'information_schema', 
            'pg_catalog', 
            'pg_toast', 
            'pg_temp_1', 
            'pg_toast_temp_1',
            'pg_statistic',
            'pg_public',
            'public'
        )
        AND s.schema_name NOT LIKE 'pg_temp_%'
        AND s.schema_name NOT LIKE 'pg_toast_temp_%'
        ORDER BY s.schema_name;
        """
        results = await self.execute_query(query)
        return [{"name": row['schema_name'], "comment": row['comment']} for row in results]

    async def get_schema_details(self, schema: str, object_name: Optional[str] = None) -> SchemaDetails:
        """Get comprehensive schema details including tables/views and other objects"""
        
        # Get table/view information using the provided query
        tables_query = """
        WITH objs AS (
          SELECT
            c.oid,
            n.nspname AS schema,
            c.relname AS name,
            c.relkind
          FROM pg_class c
          JOIN pg_namespace n ON n.oid = c.relnamespace
          WHERE n.nspname = $1::text
            AND c.relkind IN ('r','p','m','v')   -- table, partitioned table, matview, view
            AND ($2::text IS NULL OR c.relname = $2::text)
        ),
        cm AS (
          SELECT o.oid, d.description AS object_comment
          FROM objs o
          LEFT JOIN pg_description d
            ON d.classoid='pg_class'::regclass AND d.objoid=o.oid AND d.objsubid=0
        ),
        defs AS (
          SELECT o.oid,
                 CASE WHEN o.relkind IN ('v','m') THEN pg_get_viewdef(o.oid, true) END AS definition
          FROM objs o
        ),
        upd AS (
          -- updatability info for views
          SELECT v.table_schema AS schema, v.table_name AS name,
                 (v.is_updatable = 'YES') AS is_updatable
          FROM information_schema.views v
          WHERE v.table_schema = $1::text
        )
        SELECT
          o.schema,
          o.name,
          CASE o.relkind
            WHEN 'r' THEN 'table'
            WHEN 'p' THEN 'table'
            WHEN 'm' THEN 'materialized view'
            WHEN 'v' THEN 'view'
          END AS object_type,
          cm.object_comment,
          defs.definition,
          upd.is_updatable
        FROM objs o
        LEFT JOIN cm   ON cm.oid = o.oid
        LEFT JOIN defs ON defs.oid = o.oid
        LEFT JOIN upd  ON upd.schema = o.schema AND upd.name = o.name
        ORDER BY o.schema, o.name
        """
        
        # Get other database objects (functions, procedures, sequences, types, etc.)
        other_objects_query = """
        WITH all_objects AS (
          -- Functions and procedures
          SELECT 
            p.proname AS name,
            CASE 
              WHEN p.prokind = 'f' THEN 'function'
              WHEN p.prokind = 'p' THEN 'procedure'
              WHEN p.prokind = 'a' THEN 'aggregate'
              WHEN p.prokind = 'w' THEN 'window'
              ELSE 'function'
            END AS type,
            obj_description(p.oid, 'pg_proc') AS comment
          FROM pg_proc p
          JOIN pg_namespace n ON n.oid = p.pronamespace
          WHERE n.nspname = $1::text
          
          UNION ALL
          
          -- Sequences
          SELECT 
            c.relname AS name,
            'sequence' AS type,
            obj_description(c.oid, 'pg_class') AS comment
          FROM pg_class c
          JOIN pg_namespace n ON n.oid = c.relnamespace
          WHERE n.nspname = $1::text AND c.relkind = 'S'
          
          UNION ALL
          
          -- Custom types (composite, enum, domain)
          SELECT 
            t.typname AS name,
            CASE 
              WHEN t.typtype = 'c' THEN 'composite type'
              WHEN t.typtype = 'd' THEN 'domain'
              WHEN t.typtype = 'e' THEN 'enum'
              ELSE 'type'
            END AS type,
            obj_description(t.oid, 'pg_type') AS comment
          FROM pg_type t
          JOIN pg_namespace n ON n.oid = t.typnamespace
          WHERE n.nspname = $1::text 
            AND t.typtype IN ('c', 'd', 'e')
            AND t.typname NOT LIKE '_%'  -- Exclude system types
          
          UNION ALL
          
          -- Extensions (if any)
          SELECT 
            e.extname AS name,
            'extension' AS type,
            NULL AS comment
          FROM pg_extension e
          WHERE e.extnamespace = (SELECT oid FROM pg_namespace WHERE nspname = $1::text)
        )
        SELECT name, type, comment
        FROM all_objects
        ORDER BY type, name
        """
        
        # Execute queries
        tables_results = await self.execute_query(tables_query, (schema, object_name))
        other_objects_results = await self.execute_query(other_objects_query, (schema,))
        
        # Transform results
        tables = [
            TableInfo(
                schema=row['schema'],
                name=row['name'],
                object_type=row['object_type'],
                object_comment=row['object_comment'],
                definition=row['definition'],
                is_updatable=row['is_updatable']
            )
            for row in tables_results
        ]
        
        other_objects = [
            OtherObjectInfo(
                name=row['name'],
                type=row['type'],
                comment=row['comment']
            )
            for row in other_objects_results
        ]
        
        return SchemaDetails(
            schema=schema,
            tables=tables,
            other_objects=other_objects
        )

    async def get_object_details(self, schema: str, object_name: str, object_type: str) -> ObjectDetails:
        """Get detailed information about a specific database object"""
        if object_type in ['table', 'view', 'materialized view']:
            return await self._get_table_details(schema, object_name, object_type)
        else:
            raise ValueError(f"Unsupported object type: {object_type}")

    async def _get_table_details(self, schema: str, table_name: str, object_type: str) -> ObjectDetails:
        """Get detailed information about a table, view, or materialized view using comprehensive query"""
        query = """
        WITH
        ot AS (
          SELECT lower($3) AS t,
                 lower($3) = 'table'              AS is_table,
                 lower($3) = 'materialized view'  AS is_matview,
                 lower($3) = 'view'               AS is_view
        ),
        base AS (
          SELECT c.oid AS obj_oid, n.nspname AS schema, c.relname AS obj_name, c.relkind
          FROM pg_class c
          JOIN pg_namespace n ON n.oid = c.relnamespace
          CROSS JOIN ot
          WHERE n.nspname = $1
            AND c.relname  = $2
            AND (
              (ot.is_table   AND c.relkind IN ('r','p')) OR
              (ot.is_matview AND c.relkind = 'm')        OR
              (ot.is_view    AND c.relkind = 'v')
            )
        ),
        cols AS (
          SELECT
            b.schema, b.obj_name AS object, b.relkind, a.attnum,
            a.attname AS column,
            pg_catalog.format_type(a.atttypid, a.atttypmod) AS data_type,
            t.oid AS type_oid,
            t.typtype                           AS type_kind,      -- 'b','e','d'
            NOT a.attnotnull                    AS is_nullable,
            pg_get_expr(ad.adbin, ad.adrelid)   AS column_default,
            (a.attidentity <> '')               AS is_identity,
            (a.attgenerated <> '')              AS is_generated,
            co.collname                         AS collation,
            descr.description                   AS column_comment
          FROM base b
          JOIN pg_attribute a ON a.attrelid = b.obj_oid AND a.attnum > 0 AND NOT a.attisdropped
          JOIN pg_type t      ON t.oid = a.atttypid
          LEFT JOIN pg_attrdef ad ON ad.adrelid = b.obj_oid AND ad.adnum = a.attnum
          LEFT JOIN pg_collation co ON co.oid = a.attcollation AND a.attcollation <> 0
          LEFT JOIN pg_description descr
                 ON descr.classoid = 'pg_class'::regclass
                AND descr.objoid   = b.obj_oid
                AND descr.objsubid = a.attnum
        ),
        enum_info AS (
          SELECT c.schema, c.object, c.attnum,
                 jsonb_agg(e.enumlabel ORDER BY e.enumsortorder) AS enum_values
          FROM cols c
          JOIN pg_enum e ON e.enumtypid = c.type_oid
          WHERE c.type_kind = 'e'
          GROUP BY c.schema, c.object, c.attnum
        ),
        domain_info AS (
          SELECT c.schema, c.object, c.attnum,
                 jsonb_build_object(
                   'base_type',  pg_catalog.format_type(bt.oid, t.typtypmod),
                   'not_null',   t.typnotnull,
                   'default',    COALESCE(pg_get_expr(t.typdefaultbin, 0, true), t.typdefault),
                   'checks',     COALESCE(
                                   (SELECT jsonb_agg(pg_get_constraintdef(con.oid, true) ORDER BY con.conname)
                                    FROM pg_constraint con
                                    WHERE con.contypid = t.oid AND con.contype='c'),
                                   '[]'::jsonb
                                 )
                 ) AS domain_details
          FROM cols c
          JOIN pg_type t  ON t.oid = c.type_oid AND c.type_kind = 'd'
          LEFT JOIN pg_type bt ON bt.oid = t.typbasetype
        ),
        seqs AS (
          SELECT c.schema, c.object, c.attnum, s.relname AS sequence_name
          FROM cols c
          JOIN base b ON b.schema=c.schema AND b.obj_name=c.object
          JOIN ot ON TRUE
          LEFT JOIN pg_depend d
            ON ot.is_table
           AND d.refobjid = b.obj_oid AND d.refobjsubid = c.attnum AND d.deptype = 'a'
          LEFT JOIN pg_class s ON s.oid = d.objid AND s.relkind = 'S'
        ),
        cons AS (
          SELECT
            c.schema, c.object, c.attnum,
            jsonb_agg(
              jsonb_build_object(
                'name', con.conname,
                'type', con.contype,                           -- 'p','u','f','c'
                'definition', pg_get_constraintdef(con.oid, true),
                'fk_target',
                  CASE WHEN con.contype='f' THEN
                    jsonb_build_object(
                      'schema', tgt_ns.nspname,
                      'table',  tgt.relname,
                      'columns',
                        (SELECT jsonb_agg(att2.attname ORDER BY ord)
                         FROM unnest(con.confkey) WITH ORDINALITY AS k(attnum, ord)
                         JOIN pg_attribute att2 ON att2.attrelid = tgt.oid AND att2.attnum = k.attnum)
                    )
                  ELSE NULL END
              )
              ORDER BY con.contype
            ) FILTER (WHERE con.oid IS NOT NULL) AS constraints
          FROM cols c
          JOIN base b ON b.schema=c.schema AND b.obj_name=c.object
          JOIN ot ON TRUE
          LEFT JOIN pg_constraint con
                 ON ot.is_table
                AND con.conrelid = b.obj_oid
                AND (c.attnum = ANY (con.conkey))
          LEFT JOIN pg_class tgt      ON tgt.oid = con.confrelid
          LEFT JOIN pg_namespace tgt_ns ON tgt_ns.oid = tgt.relnamespace
          GROUP BY c.schema, c.object, c.attnum
        ),
        idx AS (
          SELECT
            a.schema, a.object, a.attnum,
            jsonb_agg(
              jsonb_build_object(
                'name', i.relname,
                'is_unique', ix.indisunique,
                'is_primary', ix.indisprimary,
                'is_partial', ix.indpred IS NOT NULL,
                'position', pos.ord,
                'ddl', pg_get_indexdef(ix.indexrelid, 0, true)
              )
              ORDER BY (NOT ix.indisprimary), (NOT ix.indisunique), i.relname
            ) FILTER (WHERE i.oid IS NOT NULL) AS indexes
          FROM cols a
          JOIN base b ON b.schema=a.schema AND b.obj_name=a.object
          JOIN ot ON TRUE
          LEFT JOIN pg_index ix
                 ON (ot.is_table OR ot.is_matview)
                AND ix.indrelid = b.obj_oid
          LEFT JOIN pg_class i  ON i.oid = ix.indexrelid
          LEFT JOIN LATERAL (
            SELECT ord
            FROM unnest(ix.indkey) WITH ORDINALITY AS k(attnum, ord)
            WHERE k.attnum = a.attnum
          ) AS pos ON TRUE
          GROUP BY a.schema, a.object, a.attnum
        )
        SELECT
          c.schema,
          c.object AS name,
          CASE c.relkind WHEN 'r' THEN 'table'
                         WHEN 'p' THEN 'table'
                         WHEN 'm' THEN 'materialized view'
                         WHEN 'v' THEN 'view' END AS object_type,
          c.column,
          c.data_type,
          CASE c.type_kind WHEN 'e' THEN 'enum'
                           WHEN 'd' THEN 'domain'
                           ELSE 'base' END AS type_category,
          c.is_nullable,
          c.column_default,
          c.is_identity,
          c.is_generated,
          c.collation,
          c.column_comment,
          COALESCE(enum_info.enum_values, '[]'::jsonb) AS enum_values,
          COALESCE(domain_info.domain_details, NULL)   AS domain_details,
          COALESCE(seqs.sequence_name, NULL)           AS owned_sequence,
          COALESCE(cons.constraints, '[]'::jsonb)      AS constraints,
          COALESCE(idx.indexes,      '[]'::jsonb)      AS indexes
        FROM cols c
        LEFT JOIN enum_info   ON enum_info.schema=c.schema AND enum_info.object=c.object AND enum_info.attnum=c.attnum
        LEFT JOIN domain_info ON domain_info.schema=c.schema AND domain_info.object=c.object AND domain_info.attnum=c.attnum
        LEFT JOIN seqs        ON seqs.schema=c.schema AND seqs.object=c.object AND seqs.attnum=c.attnum
        LEFT JOIN cons        ON cons.schema=c.schema AND cons.object=c.object AND cons.attnum=c.attnum
        LEFT JOIN idx         ON idx.schema=c.schema  AND idx.object=c.object  AND idx.attnum=c.attnum
        ORDER BY c.schema, c.object, c.attnum;
        """
        
        results = await self.execute_query(query, (schema, table_name, object_type))
        
        if not results:
            raise ValueError(f"Object {table_name} not found in schema {schema}")
        
        # Transform results into list of column details (deduplicated by column_name)
        columns_data = {}
        
        for row in results:
            column_name = row["column"]
            if column_name not in columns_data:
                columns_data[column_name] = {
                    "column_name": column_name,
                    "data_type": row["data_type"],
                    "type_category": row["type_category"],
                    "is_nullable": row["is_nullable"],
                    "column_default": row["column_default"],
                    "is_identity": row["is_identity"],
                    "is_generated": row["is_generated"],
                    "collation": row["collation"],
                    "column_comment": row["column_comment"],
                    "enum_values": row["enum_values"],
                    "domain_details": row["domain_details"],
                    "owned_sequence": row["owned_sequence"],
                    "constraints": row["constraints"],
                    "indexes": row["indexes"]
                }
        
        # Convert dict values back to list
        columns_list = list(columns_data.values())
        
        return ObjectDetails(
            name=table_name,
            type=object_type,
            schema=schema,
            columns=columns_list,
            constraints=[],  # Constraints are now per-column
            comments=None
        )


    async def execute_sql(self, sql: str) -> Dict[str, Any]:
        """Execute arbitrary SQL and return results"""
        try:
            results = await self.execute_query(sql)
            return {
                "status": "success",
                "rows_affected": len(results),
                "data": results
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "rows_affected": 0,
                "data": []
            }


# Initialize the MCP server
mcp = FastMCP("postgres", host=MCP_HOST, port=MCP_PORT)

# Initialize the PostgreSQL client
db_client = PostgreSQLClient()

# BLOG: Explain how to make this scalable by using caching
@mcp.tool()
async def list_schemas(ctx: Context) -> MCPResponse:
    """
    List all user-defined schemas in the PostgreSQL database with their comments
    
    Returns:
        List of user-defined schemas with name and comment information (excludes PostgreSQL system schemas)
    """
    try:
        await log("Listing database schemas with comments", "info", logger, ctx)
        schemas = await db_client.list_schemas()
        await log(f"Found {len(schemas)} schemas", "info", logger, ctx)
        return MCPResponse(
            status="OK", 
            payload={
                "message": f"Found {len(schemas)} schemas",
                "schema_list": schemas
            }
        )
    except Exception as e:
        await log(f"Error listing schemas: {str(e)}", "error", logger, ctx, exception=e)
        return MCPResponse(status="ERR", error=str(e))


@mcp.tool()
async def get_schema_details(schema: str, object_name: Optional[str] = None, ctx: Context = None) -> MCPResponse:
    """
    Get comprehensive schema details including tables/views and other database objects
    
    Args:
        schema: The schema name to get details from (e.g., 'public', 'app', 'analytics')
        object_name: Optional specific object name to filter by (if provided, only that object will be returned)
        
    Returns:
        Comprehensive schema information including:
        - Tables: name, type (table/view/materialized view), comments, definitions (for views), updatability (for views)
        - Other objects: functions, procedures, sequences, custom types (enum/domain/composite), extensions
        Each object includes name, type, and optional comments
    """
    try:
        await log(f"Getting schema details for {schema}" + (f" (filtering by {object_name})" if object_name else ""), "info", logger, ctx)
        schema_details = await db_client.get_schema_details(schema, object_name)
        await log(f"Retrieved schema details: {len(schema_details.tables)} tables, {len(schema_details.other_objects)} other objects", "info", logger, ctx)
        
        # Convert to dict format for JSON serialization
        schema_data = {
            "schema": schema_details.schema,
            "tables": [
                {
                    "name": table.name,
                    "type": table.object_type,
                    "comment": table.object_comment,
                    "definition": table.definition,
                    "is_updatable": table.is_updatable
                }
                for table in schema_details.tables
            ],
            "other_objects": [
                {
                    "name": obj.name,
                    "type": obj.type,
                    "comment": obj.comment
                }
                for obj in schema_details.other_objects
            ]
        }
        
        return MCPResponse(
            status="OK", 
            payload={
                "message": f"Retrieved schema details for {schema}",
                "schema_details": schema_data
            }
        )
    except Exception as e:
        await log(f"Error getting schema details for {schema}: {str(e)}", "error", logger, ctx, exception=e)
        return MCPResponse(status="ERR", error=str(e))


@mcp.tool()
async def list_object_details(
    schema: str, 
    object_name: str, 
    object_type: Literal["table", "view", "materialized view"], 
    ctx: Context
) -> MCPResponse:
    """
    Get comprehensive information about a specific database object returning detailed column information
    
    Args:
        schema: The schema name containing the object (e.g., 'public', 'app', 'analytics')
        object_name: The name of the database object (table, view, or materialized view)
        object_type: The type of object - must be one of: 'table', 'view', or 'materialized view'
        
    Returns:
        List of detailed column information, where each column includes:
        - Basic info: column_name, data_type, type_category (base/enum/domain), is_nullable, column_default
        - Advanced features: is_identity, is_generated, collation, column_comment
        - Type-specific data: enum_values (for enum types), domain_details (for domain types)
        - Relationships: owned_sequence (for identity columns)
        - Constraints: Primary keys, foreign keys, unique constraints, check constraints with definitions
        - Indexes: All indexes containing this column with uniqueness, primary key status, and DDL
    """
    try:
        await log(f"Getting details for {object_type} {schema}.{object_name}", "info", logger, ctx)
        details = await db_client.get_object_details(schema, object_name, object_type)
        await log(f"Retrieved details for {object_type} {schema}.{object_name}", "info", logger, ctx)
        
        # Convert to dict format for JSON serialization
        details_data = {
            "name": details.name,
            "type": details.type,
            "schema": details.schema,
            "columns": details.columns,
            "constraints": details.constraints,
            "comments": details.comments
        }
        
        return MCPResponse(
            status="OK", 
            payload={
                "message": f"Retrieved details for {object_type} {schema}.{object_name}",
                "object_details": details_data
            }
        )
    except Exception as e:
        await log(f"Error getting details for {object_type} {schema}.{object_name}: {str(e)}", "error", logger, ctx, exception=e)
        return MCPResponse(status="ERR", error=str(e))


@mcp.tool()
async def execute_sql(sql: str, ctx: Context) -> MCPResponse:
    """
    Execute arbitrary SQL query on the PostgreSQL database with full error handling and parameterization
    
    Args:
        sql: The SQL query to execute (supports SELECT, INSERT, UPDATE, DELETE, CREATE, DROP, ALTER, etc.)
        
    Returns:
        Query execution results including:
        - Status: 'success' or 'error'
        - Rows affected: Number of rows returned or modified
        - Data: Array of result rows (for SELECT queries) or empty array
        - Error message: Detailed error information if execution failed
        
    Examples:
        - SELECT * FROM users WHERE id = 123
        - INSERT INTO logs (message, level) VALUES ("error occurred", "error")
        - CREATE TABLE test (id SERIAL PRIMARY KEY, name TEXT)
    """
    try:
        await log(f"Executing SQL query: {sql[:100]}...", "info", logger, ctx)
        
        result = await db_client.execute_sql(sql)
        
        if result["status"] == "success":
            await log(f"SQL query executed successfully, {result['rows_affected']} rows returned", "info", logger, ctx)
        else:
            await log(f"SQL query failed: {result['error']}", "error", logger, ctx)
        
        return MCPResponse(
            status="OK", 
            payload={
                "message": f"SQL query executed successfully, {result['rows_affected']} rows returned" if result["status"] == "success" else f"SQL query failed: {result['error']}",
                "query_result": result
            }
        )
    except Exception as e:
        await log(f"Error executing SQL: {str(e)}", "error", logger, ctx, exception=e)
        return MCPResponse(status="ERR", error=str(e))


async def main():
    """Main function to start the PostgreSQL MCP server"""
    def log_info():
        # Mask password in connection string for logging
        masked_connection = DB_CONNECTION_STRING
        if "@" in DB_CONNECTION_STRING and ":" in DB_CONNECTION_STRING:
            # Replace password with *** for security
            parts = DB_CONNECTION_STRING.split("@")
            if len(parts) == 2:
                user_part = parts[0]
                if ":" in user_part:
                    user, password = user_part.rsplit(":", 1)
                    masked_connection = f"{user}:***@{parts[1]}"
        
        logger.info(f"PostgreSQL connection: {masked_connection}")
    
    await start_mcp_server(mcp, MCP_HOST, MCP_PORT, logger, log_info)


if __name__ == "__main__":
    asyncio.run(main())
