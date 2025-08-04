from pathlib import Path
from typing import List

from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter


class DocParser:

    def __init__(self):
        return

    def parse(self):

        documents = self.find_documents()

        doc_converter = DocumentConverter(  # all of the below is optional, has internal defaults.
            allowed_formats=[
                InputFormat.PDF,
            ],  # whitelist formats, non-matching files are ignored.
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_cls=StandardPdfPipeline, backend=PyPdfiumDocumentBackend),
            },
        )

        conv_results = doc_converter.convert_all(documents)
        md_outputs = [res.document.export_to_markdown() for res in conv_results]

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )

        chunked_outputs = []
        for md_output in md_outputs:
            chunks = text_splitter.split_text(md_output)
            chunked_outputs.extend(chunks)

        return chunked_outputs

    def find_documents(self) -> List[Path]:
        base_path = Path("./data/raw")
        return [p for p in base_path.rglob("*") if p.suffix in [".pdf"]]
