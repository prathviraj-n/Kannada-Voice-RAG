from src.ingest_pdf import process_pdf
from src.build_faiss_index import build_index_for_pdf, build_e5_index_for_pdf

for filename, base in [('Kan-Science.pdf', 'Kan-Science'),
                        ('Kannada-EVS-5-Part1.pdf', 'Kannada-EVS-5-Part1')]:
    print(f'=== {filename} ===')
    process_pdf(filename)
    build_index_for_pdf(base)
    build_e5_index_for_pdf(base)
