from io import BytesIO

import PyPDF2


def extract_text_from_pdf(pdf_bytes: BytesIO) -> str:
    """
    Extract text from a PDF file.

    Parameters
    ----------
    pdf_bytes : BytesIO
        The PDF file as a BytesIO object.

    Returns
    -------
    pdf_text : str
        The extracted text from the PDF file.

    Raises
    ------
    ValueError
        If the provided PDF file is empty or invalid.

    Notes
    -----
    This function uses the PyPDF2 library to extract text from a PDF file.
    It assumes that the PDF file is in a valid format and can be read.

    Examples
    --------
    >>> from io import BytesIO
    >>> pdf_bytes = BytesIO(b'%PDF-1.7\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n3 0 obj\n<< /Type /Page /Parent 2 0 R /Contents 4 0 R >>\nendobj\n4 0 obj\n<< /Length 27 >>\nstream\nBT\n/F1 12 Tf\n100 100 Td\n(Hello, World!) Tj\nET\nendstream\nendobj\nxref\n0 5\n0000000000 65535 f\n0000000010 00000 n\n0000000053 00000 n\n0000000103 00000 n\n0000000174 00000 n\ntrailer\n<< /Size 5 /Root 1 0 R >>\nstartxref\n227\n%%EOF\n')
    >>> extract_text_from_pdf(pdf_bytes)
    'Hello, World!'
    """
    # TODO: Use PyPDF2.PdfReader to open the input `pdf_bytes` and extract the text from each page appended to `pdf_text`.
    # Hint: Use the `extract_text()` method of the `PyPDF2.PdfReader` object.
    #warnings.filterwarnings("ignore", category=DeprecationWarning)
    pdf_text = ""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_bytes)
    except Exception as e:
        raise ValueError(f"Failed to read the PDF file: {e}")

    
    if len(pdf_reader.pages) == 0:
        raise ValueError("The provided PDF file is empty.")
    
    pdf_text = ""
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        pdf_text += page.extract_text() if page.extract_text() else ""
    
    return pdf_text

