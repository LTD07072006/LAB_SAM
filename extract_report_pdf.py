from pathlib import Path
from pypdf import PdfReader

pdf = Path(r"D:\ITK_SNAP\data\Báo_cáo_tuần_8 (1).pdf")
reader = PdfReader(str(pdf))
print("pages", len(reader.pages))
for index, page in enumerate(reader.pages, start=1):
    print(f"---PAGE {index}---")
    print(page.extract_text() or "")
