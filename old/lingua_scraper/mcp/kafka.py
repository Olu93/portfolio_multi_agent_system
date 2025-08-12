from mcp.server.fastmcp import FastMCP, Context
from kafka import KafkaProducer, KafkaConsumer, KafkaAdminClient
from kafka.admin import NewTopic, ConfigResource, ConfigResourceType
from kafka.errors import TopicAlreadyExistsError, UnknownTopicOrPartitionError, NoBrokersAvailable
from typing import List, Dict, Optional, Any
import json
import os
import sys
import traceback
import asyncio
from datetime import datetime, timedelta
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())


# Get configuration from environment variables
MCP_HOST = os.getenv("MCP_HOST", "localhost")
MCP_PORT = int(os.getenv("MCP_PORT", "8004"))
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")


class KafkaManager:
    """Manages Kafka operations including topic management and message handling"""
    
    def __init__(self):
        self.bootstrap_servers = KAFKA_BOOTSTRAP_SERVERS
        self.admin_client = None
        self.producer = None
        self.consumer = None
        
    async def get_admin_client(self, ctx: Context) -> KafkaAdminClient:
        """Get or create Kafka admin client"""
        try:
            if self.admin_client is None:
                await ctx.info(f"Connecting to Kafka admin client at: {self.bootstrap_servers}")
                self.admin_client = KafkaAdminClient(
                    bootstrap_servers=self.bootstrap_servers,
                    client_id='mcp-kafka-admin'
                )
                await ctx.info("Successfully connected to Kafka admin client")
            return self.admin_client
        except NoBrokersAvailable:
            await ctx.error(f"Could not connect to Kafka brokers at {self.bootstrap_servers}")
            raise
        except Exception as e:
            await ctx.error(f"Error connecting to Kafka admin client: {str(e)}")
            raise
    
    async def get_producer(self, ctx: Context) -> KafkaProducer:
        """Get or create Kafka producer"""
        try:
            if self.producer is None:
                await ctx.info(f"Connecting to Kafka producer at: {self.bootstrap_servers}")
                self.producer = KafkaProducer(
                    bootstrap_servers=self.bootstrap_servers,
                    value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                    key_serializer=lambda k: k.encode('utf-8') if k else None,
                    client_id='mcp-kafka-producer'
                )
                await ctx.info("Successfully connected to Kafka producer")
            return self.producer
        except NoBrokersAvailable:
            await ctx.error(f"Could not connect to Kafka brokers at {self.bootstrap_servers}")
            raise
        except Exception as e:
            await ctx.error(f"Error connecting to Kafka producer: {str(e)}")
            raise
    
    async def create_consumer(self, topic: str, group_id: str, ctx: Context) -> KafkaConsumer:
        """Create a new Kafka consumer"""
        try:
            await ctx.info(f"Creating Kafka consumer for topic: {topic}")
            consumer = KafkaConsumer(
                topic,
                bootstrap_servers=self.bootstrap_servers,
                group_id=group_id,
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                key_deserializer=lambda k: k.decode('utf-8') if k else None,
                auto_offset_reset='earliest',
                enable_auto_commit=True,
                client_id='mcp-kafka-consumer'
            )
            await ctx.info("Successfully created Kafka consumer")
            return consumer
        except NoBrokersAvailable:
            await ctx.error(f"Could not connect to Kafka brokers at {self.bootstrap_servers}")
            raise
        except Exception as e:
            await ctx.error(f"Error creating Kafka consumer: {str(e)}")
            raise
    
    async def close_connections(self):
        """Close all Kafka connections"""
        try:
            if self.admin_client:
                self.admin_client.close()
            if self.producer:
                self.producer.close()
            if self.consumer:
                self.consumer.close()
        except Exception as e:
            print(f"Error closing Kafka connections: {str(e)}")


# Initialize the MCP server
mcp = FastMCP("kafka", host=MCP_HOST, port=MCP_PORT)

# Initialize the Kafka manager
kafka_manager = KafkaManager()


@mcp.tool()
async def create_topic(
    topic_name: str,
    num_partitions: int = 1,
    replication_factor: int = 1,
    ctx: Context = None
) -> str:
    """
    Create a new Kafka topic
    
    Args:
        topic_name: Name of the topic to create
        num_partitions: Number of partitions for the topic (default: 1)
        replication_factor: Replication factor for the topic (default: 1)
        ctx: MCP context for logging
    """
    try:
        admin_client = await kafka_manager.get_admin_client(ctx)
        
        await ctx.info(f"Creating topic '{topic_name}' with {num_partitions} partitions and replication factor {replication_factor}")
        
        new_topic = NewTopic(
            name=topic_name,
            num_partitions=num_partitions,
            replication_factor=replication_factor
        )
        
        admin_client.create_topics([new_topic])
        
        await ctx.info(f"Successfully created topic '{topic_name}'")
        return f"Topic '{topic_name}' created successfully with {num_partitions} partitions and replication factor {replication_factor}"
        
    except TopicAlreadyExistsError:
        await ctx.error(f"Topic '{topic_name}' already exists")
        return f"Topic '{topic_name}' already exists"
    except Exception as e:
        await ctx.error(f"Error creating topic '{topic_name}': {str(e)}")
        return f"Error creating topic '{topic_name}': {str(e)}"


@mcp.tool()
async def list_topics(ctx: Context = None) -> str:
    """
    List all available Kafka topics
    
    Args:
        ctx: MCP context for logging
    """
    try:
        admin_client = await kafka_manager.get_admin_client(ctx)
        
        await ctx.info("Fetching list of Kafka topics")
        
        metadata = admin_client.list_topics()
        topics = list(metadata)
        
        await ctx.info(f"Found {len(topics)} topics")
        
        if topics:
            result = f"Found {len(topics)} topics:\n"
            for topic in sorted(topics):
                result += f"  - {topic}\n"
            return result
        else:
            return "No topics found in the Kafka cluster"
            
    except Exception as e:
        await ctx.error(f"Error listing topics: {str(e)}")
        return f"Error listing topics: {str(e)}"


@mcp.tool()
async def delete_topic(topic_name: str, ctx: Context = None) -> str:
    """
    Delete an existing Kafka topic
    
    Args:
        topic_name: Name of the topic to delete
        ctx: MCP context for logging
    """
    try:
        admin_client = await kafka_manager.get_admin_client(ctx)
        
        await ctx.info(f"Deleting topic '{topic_name}'")
        
        admin_client.delete_topics([topic_name])
        
        await ctx.info(f"Successfully deleted topic '{topic_name}'")
        return f"Topic '{topic_name}' deleted successfully"
        
    except UnknownTopicOrPartitionError:
        await ctx.error(f"Topic '{topic_name}' does not exist")
        return f"Topic '{topic_name}' does not exist"
    except Exception as e:
        await ctx.error(f"Error deleting topic '{topic_name}': {str(e)}")
        return f"Error deleting topic '{topic_name}': {str(e)}"


@mcp.tool()
async def describe_topic(topic_name: str, ctx: Context = None) -> str:
    """
    Get detailed information about a specific Kafka topic
    
    Args:
        topic_name: Name of the topic to describe
        ctx: MCP context for logging
    """
    try:
        admin_client = await kafka_manager.get_admin_client(ctx)
        
        await ctx.info(f"Describing topic '{topic_name}'")
        
        # Get topic metadata
        metadata = admin_client.list_topics()
        if topic_name not in metadata:
            await ctx.error(f"Topic '{topic_name}' does not exist")
            return f"Topic '{topic_name}' does not exist"
        
        # Get topic configuration
        config_resource = ConfigResource(ConfigResourceType.TOPIC, topic_name)
        configs = admin_client.describe_configs([config_resource])
        
        # Get partition information
        cluster_metadata = admin_client.list_topics()
        topic_metadata = cluster_metadata[topic_name]
        
        result = f"Topic: {topic_name}\n"
        result += f"Partitions: {len(topic_metadata.partitions)}\n"
        result += f"Replication Factor: {topic_metadata.replication_factor}\n"
        result += f"Partition Details:\n"
        
        for partition_id, partition_metadata in topic_metadata.partitions.items():
            result += f"  Partition {partition_id}:\n"
            result += f"    Leader: {partition_metadata.leader}\n"
            result += f"    Replicas: {partition_metadata.replicas}\n"
            result += f"    ISR: {partition_metadata.isr}\n"
        
        await ctx.info(f"Successfully described topic '{topic_name}'")
        return result
        
    except Exception as e:
        await ctx.error(f"Error describing topic '{topic_name}': {str(e)}")
        return f"Error describing topic '{topic_name}': {str(e)}"


@mcp.tool()
async def produce_message(
    topic_name: str,
    message: str,
    key: Optional[str] = None,
    headers: Optional[str] = None,
    ctx: Context = None
) -> str:
    """
    Send a message to a Kafka topic
    
    Args:
        topic_name: Name of the topic to send message to
        message: Message content to send
        key: Optional message key
        headers: Optional JSON string of headers
        ctx: MCP context for logging
    """
    try:
        producer = await kafka_manager.get_producer(ctx)
        
        await ctx.info(f"Producing message to topic '{topic_name}'")
        
        # Parse headers if provided
        message_headers = []
        if headers:
            try:
                headers_dict = json.loads(headers)
                message_headers = [(k, v.encode('utf-8')) for k, v in headers_dict.items()]
            except json.JSONDecodeError:
                await ctx.error("Invalid JSON format for headers")
                return "Error: Invalid JSON format for headers"
        
        # Send message
        future = producer.send(
            topic_name,
            value=message,
            key=key,
            headers=message_headers
        )
        
        # Wait for the message to be sent
        record_metadata = future.get(timeout=10)
        
        await ctx.info(f"Successfully produced message to topic '{topic_name}'")
        return f"Message sent successfully to topic '{topic_name}' at partition {record_metadata.partition}, offset {record_metadata.offset}"
        
    except Exception as e:
        await ctx.error(f"Error producing message to topic '{topic_name}': {str(e)}")
        return f"Error producing message to topic '{topic_name}': {str(e)}"


@mcp.tool()
async def consume_messages(
    topic_name: str,
    group_id: str = "mcp-consumer-group",
    timeout_ms: int = 5000,
    max_messages: int = 10,
    ctx: Context = None
) -> str:
    """
    Consume messages from a Kafka topic
    
    Args:
        topic_name: Name of the topic to consume from
        group_id: Consumer group ID
        timeout_ms: Timeout in milliseconds for polling
        max_messages: Maximum number of messages to consume
        ctx: MCP context for logging
    """
    try:
        consumer = await kafka_manager.create_consumer(topic_name, group_id, ctx)
        
        await ctx.info(f"Consuming messages from topic '{topic_name}' (max: {max_messages})")
        
        messages = []
        message_count = 0
        
        # Poll for messages
        while message_count < max_messages:
            msg_pack = consumer.poll(timeout_ms=timeout_ms)
            
            if not msg_pack:
                break
                
            for tp, records in msg_pack.items():
                for record in records:
                    if message_count >= max_messages:
                        break
                        
                    message_info = {
                        "topic": record.topic,
                        "partition": record.partition,
                        "offset": record.offset,
                        "key": record.key,
                        "value": record.value,
                        "timestamp": record.timestamp,
                        "headers": dict(record.headers) if record.headers else {}
                    }
                    messages.append(message_info)
                    message_count += 1
        
        consumer.close()
        
        await ctx.info(f"Successfully consumed {len(messages)} messages from topic '{topic_name}'")
        
        if messages:
            result = f"Consumed {len(messages)} messages from topic '{topic_name}':\n\n"
            for i, msg in enumerate(messages, 1):
                result += f"Message {i}:\n"
                result += f"  Partition: {msg['partition']}\n"
                result += f"  Offset: {msg['offset']}\n"
                result += f"  Key: {msg['key']}\n"
                result += f"  Value: {msg['value']}\n"
                result += f"  Timestamp: {msg['timestamp']}\n"
                if msg['headers']:
                    result += f"  Headers: {msg['headers']}\n"
                result += "\n"
            return result
        else:
            return f"No messages consumed from topic '{topic_name}' within timeout"
            
    except Exception as e:
        await ctx.error(f"Error consuming messages from topic '{topic_name}': {str(e)}")
        return f"Error consuming messages from topic '{topic_name}': {str(e)}"


@mcp.tool()
async def get_topic_partition_count(topic_name: str, ctx: Context = None) -> str:
    """
    Get the number of partitions for a specific topic
    
    Args:
        topic_name: Name of the topic
        ctx: MCP context for logging
    """
    try:
        admin_client = await kafka_manager.get_admin_client(ctx)
        
        await ctx.info(f"Getting partition count for topic '{topic_name}'")
        
        metadata = admin_client.list_topics()
        if topic_name not in metadata:
            await ctx.error(f"Topic '{topic_name}' does not exist")
            return f"Topic '{topic_name}' does not exist"
        
        topic_metadata = metadata[topic_name]
        partition_count = len(topic_metadata.partitions)
        
        await ctx.info(f"Topic '{topic_name}' has {partition_count} partitions")
        return f"Topic '{topic_name}' has {partition_count} partitions"
        
    except Exception as e:
        await ctx.error(f"Error getting partition count for topic '{topic_name}': {str(e)}")
        return f"Error getting partition count for topic '{topic_name}': {str(e)}"


@mcp.tool()
async def wait_before_trying_again(seconds: int, ctx: Context = None) -> str:
    """
    Wait for a specified number of seconds before trying again
    
    Args:
        seconds: Number of seconds to wait
        ctx: MCP context for logging
    """
    if ctx:
        await ctx.info(f"Waiting for {seconds} seconds...")
    await asyncio.sleep(seconds)
    return f"Waited for {seconds} seconds. You can now try your operation again."


if __name__ == "__main__":
    print("=== Starting Kafka MCP Server ===")
    print(f"Server will run on {MCP_HOST}:{MCP_PORT}")
    print(f"Kafka bootstrap servers: {KAFKA_BOOTSTRAP_SERVERS}")
    try:
        print("Server initialized and ready to handle connections")
        mcp.run(transport="streamable-http")
    except Exception as e:
        print(f"Server crashed: {str(e)}", exc_info=True)
        raise
    finally:
        print("=== Kafka MCP Server shutting down ===")
        asyncio.run(kafka_manager.close_connections()) 