"""
Parses PDF, Markdown, and HTML files using LangChain document loaders.
"""

from pathlib import Path
from langchain_community.document_loaders import BSHTMLLoader, PyPDFLoader
from langchain_community.document_loaders.markdown import UnstructuredMarkdownLoader
from langchain_core.documents import Document

# from pypdf import PdfReader
# from bs4 import BeautifulSoup
# import markdown
# from dataclasses import dataclass
#
# @dataclass
# class ParsedDocument:
#     content: str
#     filename: str
#     doc_type: str
#     page_count: int | None = None
#
# def parse_pdf(filepath: str) -> ParsedDocument:
#     reader = PdfReader(filepath)
#     pages = [page.extract_text() or "" for page in reader.pages]
#     return ParsedDocument(
#         content="\n\n".join(pages),
#         filename=Path(filepath).name,
#         doc_type="pdf",
#         page_count=len(reader.pages),
#     )
#
# def parse_markdown(filepath: str) -> ParsedDocument:
#     text = Path(filepath).read_text(encoding="utf-8")
#     html = markdown.markdown(text)
#     clean = BeautifulSoup(html, "html.parser").get_text(separator="\n")
#     return ParsedDocument(
#         content=clean,
#         filename=Path(filepath).name,
#         doc_type="markdown",
#     )
#
# def parse_html(filepath: str) -> ParsedDocument:
#     html = Path(filepath).read_text(encoding="utf-8")
#     clean = BeautifulSoup(html, "html.parser").get_text(separator="\n")
#     return ParsedDocument(
#         content=clean,
#         filename=Path(filepath).name,
#         doc_type="html",
#     )


def parse_pdf(filepath: str) -> list[Document]:
    return PyPDFLoader(filepath).load()


def parse_markdown(filepath: str) -> list[Document]:
    return UnstructuredMarkdownLoader(filepath).load()


def parse_html(filepath: str) -> list[Document]:
    return BSHTMLLoader(filepath).load()


def parse_document(filepath: str) -> list[Document]:
    """Route to the right loader based on file extension."""
    ext = Path(filepath).suffix.lower()
    parsers = {
        ".pdf": parse_pdf,
        ".md": parse_markdown,
        ".html": parse_html,
        ".htm": parse_html,
    }
    parser = parsers.get(ext)
    if not parser:
        raise ValueError(f"Unsupported file type: {ext}")
    return parser(filepath)
