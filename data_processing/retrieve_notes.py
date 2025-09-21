import os
import re
from langchain_community.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter, TextSplitter
from langchain.docstore.document import Document
import markdown2
import glob
from langchain_text_splitters import  HTMLSemanticPreservingSplitter, MarkdownHeaderTextSplitter, CharacterTextSplitter

import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

import sqlite3
DATABASE_URL = "sqlite:///./data/test.db"

FILES_SEEN_PATH = "data/files_seen.txt"

# Regex to match date pattern
date_regex = re.compile(r"(\d{2})\s*([A-Za-z]{3})\s*(\d{2})")
month_map = {m: i+1 for i, m in enumerate(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])}

conn = sqlite3.connect("data/test.db")
c = conn.cursor()


def load_files_seen():
    if not os.path.exists(FILES_SEEN_PATH):
        return set()
    with open(FILES_SEEN_PATH, "r") as f:
        return set(line.strip() for line in f if line.strip())

def save_files_seen(files_seen):
    with open(FILES_SEEN_PATH, "w") as f:
        for file in sorted(files_seen):
            f.write(file + "\n")


def load_markdown_files_naively(root_folder: str):
    docs = []
    for dirpath, _, filenames in os.walk(root_folder):
        for file in filenames:
            if file.endswith(".md"):
                filepath = os.path.join(dirpath, file)
                with open(filepath, "r", encoding="utf-8") as f:
                    text = f.read()
                    html = markdown2.markdown(text)
                    docs.append(Document(page_content=text, metadata={"source": filepath}))
    return docs

def parse_date(source):
    match = date_regex.search(source)
    if match:
        year = int(match.group(1))
        month = month_map.get(match.group(2).capitalize(), 1)
        day = int(match.group(3))
        # Assume year 2000+
        year += 2000
        return year, month, day, int(f"{year:04d}{month:02d}{day:02d}")
    else:
        # Very old date for non-matching sources
        return 1900, 1, 1, 19000101

def is_important(source):
    return 1 if "@@" in source else 0

def load_markdown_files_naively_semantically_preserved(filepaths: list, files_seen: set):
    
    # headers_to_split_on = [
    #     ("h1", "Header 1"),
    #     ("h2", "Header 2"),
    #     ("h3", "Header 3"),
    #     ("h4", "Header 4"),
    #     ("h5", "Header 5"),
    #     ("h6", "Header 6")]
    
    # def code_handler(element: Tag) -> str:
    #     data_lang = element.get("data-lang")
    #     code_format = f"<code:{data_lang}>{element.get_text()}</code>"

    #     return code_format
    
    # splitter = HTMLSemanticPreservingSplitter(
    #     headers_to_split_on=headers_to_split_on,
    #     separators=["\n\n", "\n", "<br />"],
    #     max_chunk_size=500,
    #     preserve_images=True,
    #     preserve_videos=True,
    #     elements_to_preserve=["table", "ul", "ol", "code"],
    #     denylist_tags=["script", "style", "head"],
    #     # custom_handlers={"code": code_handler},
    # )

    # headers_to_split_on = [
    #     ("#", "Header 1"),
    #     ("##", "Header 2"),
    #     ("###", "Header 3"),
    #     ("####", "Header 4"),
    #     ("#####", "Header 5"),
    #     ("######", "Header 6")
    # ]

    # splitter = MarkdownHeaderTextSplitter(
    #     headers_to_split_on,
    #     return_each_line=True,
    #     strip_headers=False
    # )


    splitter = CharacterTextSplitter(
        separator="\n\n",
        chunk_size=1000,
        length_function=len,
        is_separator_regex=False,
    )
    total_dates = 0
    docs = []
    
    for filepath in filepaths:
        file = os.path.basename(filepath)

        if file not in files_seen:
            files_seen.add(file)

        with open(filepath, "r", encoding="utf-8") as f:
            text_original = f.read()
            text = text_original.replace("\n  \n  \n  \n  \n  \n", "\n\n").replace("\n  \n  \n  \n  \n", "\n\n").replace("\n  \n  \n  \n", "\n\n").replace("\n  \n  \n", "\n\n").replace("\n  \n", "\n\n")
            # html = markdown2.markdown(text)
            # documents = splitter.split_text(text)
            # Update metadata for each document
            # for i, document in enumerate(documents):
            #     topics = extract_topics_from_entry(document.page_content)
            #     document.metadata["topics"] = topics
            
            contents = text.split("\n\n")
            # documents = [Document(page_content=content.strip(), metadata={}) for content in contents if len(content.strip()) > 0]

            documents = []
            for num, content in enumerate(contents):
                if len(content.strip()) > 0:
                    
                    date_header_regex = r"#{1,4} {1,4}\*?\*?(\d{2}|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec){1}.*"
                    if re.match(date_header_regex, content.split('\n')[0].lower().strip()):
                        print(f"Skipping date header: {content.strip()}")
                        total_dates += 1
                        continue
                        
                    
                    # Create a Document object for each content
                    context = [contents[i] for i in range(num-1 if num-1 >= 0 else 0, 
                                                        num+2 if num+2< len(contents) else len(contents)) 
                                                        if len(contents[i].strip())>0]
                    context = "\n\n".join(context)

                    doc = Document(page_content=content.strip(), 
                                metadata={
                                    "source": filepath.split("/")[-1].split(".")[0],
                                    "context": context
                                    }
                                    )
                    documents.append(doc)

            docs.extend(documents)
            if len(docs) % 500 == 0:
                print(f"Processed {len(docs)} documents so far."    )


    print(f"Total date headers skipped: {total_dates}")
    return docs, files_seen

def retrieve_notes():

    files_seen = load_files_seen()

    print("🔍 Loading documents...")
    filepaths = glob.glob("../PersonalNotes/**/*.md", recursive=True)
    docs, update_files_seen = load_markdown_files_naively_semantically_preserved(filepaths, files_seen)
    print(f"Loaded {len(docs)} markdown files.")

    return docs



if __name__ == "__main__":
    retrieve_notes()
