import io
import fitz
import app


def main():
    pdf_bytes = io.BytesIO()
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), 'This is a sample PDF for testing semantic plagiarism detection.')
    doc.save(pdf_bytes)
    doc.close()

    text = app.extract_text_from_pdf(pdf_bytes.getvalue())
    print('PDF_OK' if 'sample pdf' in text.lower() else 'PDF_FAIL')
    print(text[:80])


if __name__ == '__main__':
    main()
