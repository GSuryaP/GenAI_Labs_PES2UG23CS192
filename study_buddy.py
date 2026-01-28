from transformers import pipeline
from PyPDF2 import PdfReader


def extract_clean_text(pdf_path):
    reader = PdfReader(pdf_path)
    cleaned_text = ""

    for page in reader.pages:
        text = page.extract_text()
        if not text:
            continue

        for line in text.split("\n"):
            line = line.strip()

            if len(line) < 40:
                continue
            if "figure" in line.lower():
                continue
            if "©" in line:
                continue

            cleaned_text += line + " "

    return cleaned_text


def chunk_text(text, chunk_size=350):
    words = text.split()
    for i in range(0, len(words), chunk_size):
        yield " ".join(words[i:i + chunk_size])

qa_pipeline = pipeline(
    "question-answering",
    model="distilbert-base-cased-distilled-squad"
)


pdf_text = extract_clean_text("pdfs/chapter1.pdf")


print("\nStudy Buddy (PDF Quizzer)")
print("Type 'exit' to quit\n")

while True:
    question = input("Ask a question: ").strip()

    if question.lower() == "exit":
        print("Goodbye!")
        break

    best_answer = ""
    best_score = 0

    for chunk in chunk_text(pdf_text):
        result = qa_pipeline(
            question=question,
            context=chunk
        )

        if result["score"] > best_score:
            best_score = result["score"]
            best_answer = result["answer"]

    if best_score < 0.15:
        print("Answer: Not found in the document.")
    else:
        if len(best_answer.split()) < 4:
            print("Answer: Definition not explicitly stated in the document.")
        else:
            print("Answer:", best_answer)
    print("-*" * 25)


