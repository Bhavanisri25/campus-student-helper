# Campus Student Helper – RAG Application

A Python-based student assistant that uses **Retrieval-Augmented Generation (RAG)** to answer student queries using campus guidelines.

## Features

- Answers student queries using campus information.
- Retrieves relevant information from the campus guidelines.
- Uses embeddings and vector search to find relevant content.
- Uses an AI model to generate answers from the retrieved information.
- Provides a simple web interface using Flask.

## Technologies Used

- Python
- Flask
- LangChain
- ChromaDB
- Hugging Face Embeddings
- Groq API
- RAG

## How It Works

1. The campus guidelines are loaded from a text file.
2. The text is divided into smaller chunks.
3. The chunks are converted into embeddings.
4. ChromaDB stores the embeddings and performs similarity search.
5. When a student asks a question, relevant information is retrieved.
6. The retrieved information is given to the AI model.
7. The AI generates the final answer.

## Project Structure

```text
campus-student-helper/
│
├── app.py
├── campus_guidelines.txt
├── requirements.txt
├── README.md
├── .gitignore
│
└── templates/
    └── index.html
```

## Installation

Clone the repository:

```bash
git clone https://github.com/Bhavanisri25/campus-student-helper.git
```

Go to the project folder:

```bash
cd campus-student-helper
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## API Key Setup

Create a `.env` file in the project folder:

```text
GROQ_API_KEY=your_api_key_here
```

Do not upload the `.env` file to GitHub.

## Run the Application

Run:

```bash
python app.py
```

Open the application in your browser:

```text
http://127.0.0.1:5000
```

## Purpose

The project demonstrates how **RAG, vector search, embeddings, and Generative AI** can be combined to build a student information assistant.

## Author

**Yatham Bhavani Sri**

GitHub: https://github.com/Bhavanisri25
