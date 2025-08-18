// src/components/DropzoneArea.js
import React, { useState } from "react";
import { useDropzone } from "react-dropzone";
import { Box, Typography, List, ListItem, ListItemText } from "@mui/material";
import Grid from "@mui/material/Grid2"; // Updated import as requested
import { Document, Page } from "react-pdf";

import { pdfjs } from "react-pdf";
import "react-pdf/dist/Page/AnnotationLayer.css";
import "react-pdf/dist/Page/TextLayer.css";

pdfjs.GlobalWorkerOptions.workerSrc = new URL(
  "pdfjs-dist/build/pdf.worker.min.mjs",
  import.meta.url
).toString();

// TODO: Add option to allow just one file to be uploaded for the Pitchdeck Analyzer from the Founders perspective
const DropzoneArea = ({ onDrop, multiple = true }) => {
  const [files, setFiles] = useState([]);
  const [filesBase64, setFilesBase64] = useState([]);

  function getBase64(file) {
    var reader = new FileReader();
    reader.readAsDataURL(file);
    reader.onload = function () {
      setFilesBase64((filesBase64) => [...filesBase64, reader.result]);
    };
    reader.onerror = function (error) {
      console.log("Error: ", error);
    };
  }
  const onDropHandler = (acceptedFiles) => {
    if (multiple) {
      setFiles(acceptedFiles);
      onDrop(acceptedFiles);
      acceptedFiles.map((file) => getBase64(file));
      console.log(acceptedFiles);
    } else {
      const file = acceptedFiles[0];
      setFiles([file]);
      onDrop([file]);
      getBase64(file);
    }
  };

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop: onDropHandler,
    accept: {
      "application/pdf": [],
      "application/vnd.ms-powerpoint": [], // Old PowerPoint format (.ppt)
      "application/vnd.openxmlformats-officedocument.presentationml.presentation":
        [], // New PowerPoint format (.pptx)
    },
    maxFiles: multiple ? undefined : 1, // Allow only one file if multiple is false
  });

  return (
    <Grid
      {...getRootProps()}
      sx={{
        border: "2px solid #cccccc",
        borderRadius: 4,
        padding: 2,
        textAlign: "center",
        cursor: "pointer",
        backgroundColor: isDragActive ? "#292929" : "#121110",
      }}
      justifyContent="center"
      alignItems="center"
      minHeight="10vh"
      direction={"row"}
      container
    >
      <Grid size={12}>
        <input {...getInputProps()} />
        <Typography sx={{ color: "#ffffff" }}>
          {isDragActive
            ? "Drop the file here..."
            : multiple
            ? "Drag & drop some files here, or click to select files"
            : "Drag & drop a file here, or click to select a file"}
        </Typography>
      </Grid>
      <Grid container spacing={4} size={12}>
        {files.length > 0 &&
          files.map((file, index) => (
            <Grid container key={index} sx={{ height: "50%" }}>
              <Grid>
                <Document
                  file={filesBase64[index]}

                  // onLoadSuccess={onDocumentLoadSuccess}
                >
                  <Page height={200} pageNumber={1} />
                </Document>
                <Grid>
                  <Typography
                    // primary=
                    // secondary={`${file.size} bytes`}
                    sx={{ color: "#ffffff" }}
                  >
                    {file.path}
                  </Typography>
                </Grid>
              </Grid>
            </Grid>
          ))}
      </Grid>
    </Grid>
  );
};

export default DropzoneArea;
