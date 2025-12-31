---
name: pdf-processing
description: Extract and analyze content from PDF documents
version: 1.0.0
tags: [document, pdf, extraction]
---

# PDF Processing Skill

This skill enables you to work with PDF documents.

## Capabilities

1. **Extract Text**: Extract all text content from PDF files
2. **Extract Tables**: Identify and extract tabular data
3. **Summarize**: Generate summaries of PDF content
4. **Search**: Search for specific content within PDFs

## Usage Guidelines

- When a user provides a PDF file path, use the extract tools first
- For large documents, consider summarizing before detailed analysis
- Tables are returned in structured format (JSON or markdown)

## Example Interactions

User: "Extract the text from report.pdf"
- Use the pdf_extract_text tool with the file path

User: "What tables are in this PDF?"
- Use the pdf_extract_tables tool to find and display tables

User: "Summarize this research paper"
- First extract text, then provide a concise summary
