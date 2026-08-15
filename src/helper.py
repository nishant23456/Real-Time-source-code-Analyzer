import os
import shutil
import zipfile
import tempfile
import requests

from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers.language.language_parser import LanguageParser

from langchain.text_splitter import Language
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings


# =========================================================
# PATH
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

REPO_PATH = os.path.join(
    BASE_DIR,
    "repo"
)


# =========================================================
# DELETE REPOSITORY
# =========================================================

def remove_repo():

    if not os.path.exists(REPO_PATH):
        return

    print("Removing old repository...")
    print("Path:", REPO_PATH)

    def remove_readonly(func, path, exc_info):

        try:
            os.chmod(path, 0o777)
            func(path)

        except Exception:
            pass

    try:

        shutil.rmtree(
            REPO_PATH,
            onerror=remove_readonly
        )

    except Exception as e:

        raise Exception(
            f"Could not delete old repository: {e}"
        )

    # Verify deletion

    if os.path.exists(REPO_PATH):

        raise Exception(
            "Old repo folder could not be deleted."
        )

    print("Old repository deleted.")


# =========================================================
# DOWNLOAD GITHUB REPOSITORY
# =========================================================

def repo_ingestion(repo_url):

    # ---------------------------------------------
    # Remove previous repository
    # ---------------------------------------------

    remove_repo()

    # Create fresh repo directory

    os.makedirs(
        REPO_PATH,
        exist_ok=True
    )


    # ---------------------------------------------
    # Clean URL
    # ---------------------------------------------

    repo_url = repo_url.strip()
    repo_url = repo_url.rstrip("/")

    if repo_url.endswith(".git"):

        repo_url = repo_url[:-4]


    # ---------------------------------------------
    # Try MAIN branch
    # ---------------------------------------------

    zip_url = (
        repo_url +
        "/archive/refs/heads/main.zip"
    )

    print("Trying:", zip_url)

    response = requests.get(
        zip_url,
        timeout=60
    )


    # ---------------------------------------------
    # Try MASTER if MAIN doesn't exist
    # ---------------------------------------------

    if response.status_code != 200:

        zip_url = (
            repo_url +
            "/archive/refs/heads/master.zip"
        )

        print("Main branch not found.")
        print("Trying:", zip_url)

        response = requests.get(
            zip_url,
            timeout=60
        )


    # ---------------------------------------------
    # Check download
    # ---------------------------------------------

    if response.status_code != 200:

        raise Exception(
            "Could not download repository.\n"
            f"HTTP status: {response.status_code}\n"
            f"URL: {repo_url}"
        )


    # ---------------------------------------------
    # Save ZIP temporarily
    # ---------------------------------------------

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".zip"
    ) as temp_file:

        temp_file.write(
            response.content
        )

        zip_path = temp_file.name


    # ---------------------------------------------
    # Extract ZIP
    # ---------------------------------------------

    try:

        with zipfile.ZipFile(
            zip_path,
            "r"
        ) as zip_ref:

            zip_ref.extractall(
                REPO_PATH
            )

    finally:

        if os.path.exists(zip_path):

            os.remove(zip_path)


    # ---------------------------------------------
    # GitHub creates:
    #
    # repo/
    #     repository-main/
    #
    # We want:
    #
    # repo/
    #     src/
    #     Data/
    #     etc.
    # ---------------------------------------------

    extracted_items = os.listdir(
        REPO_PATH
    )


    if len(extracted_items) == 1:

        extracted_folder = os.path.join(
            REPO_PATH,
            extracted_items[0]
        )


        if os.path.isdir(
            extracted_folder
        ):

            for item in os.listdir(
                extracted_folder
            ):

                source = os.path.join(
                    extracted_folder,
                    item
                )

                destination = os.path.join(
                    REPO_PATH,
                    item
                )

                shutil.move(
                    source,
                    destination
                )


            shutil.rmtree(
                extracted_folder,
                ignore_errors=True
            )


    # ---------------------------------------------
    # IMPORTANT:
    # Remove .git if somehow present
    # ---------------------------------------------

    git_folder = os.path.join(
        REPO_PATH,
        ".git"
    )

    if os.path.exists(git_folder):

        print("Removing .git folder...")

        shutil.rmtree(
            git_folder,
            ignore_errors=True
        )


    print()
    print("=" * 60)
    print("Repository downloaded successfully.")
    print("Repository path:", REPO_PATH)
    print("=" * 60)
    print()


# =========================================================
# LOAD REPOSITORY
# =========================================================

def load_repo(repo_path):

    loader = GenericLoader.from_filesystem(
        repo_path,
        glob="**/*",
        suffixes=[".py"],
        parser=LanguageParser(
            language=Language.PYTHON,
            parser_threshold=500
        )
    )

    documents = loader.load()

    return documents


# =========================================================
# TEXT SPLITTER
# =========================================================

def text_splitter(documents):

    documents_splitter = (
        RecursiveCharacterTextSplitter.from_language(
            language=Language.PYTHON,
            chunk_size=2000,
            chunk_overlap=200
        )
    )

    text_chunks = (
        documents_splitter.split_documents(
            documents
        )
    )

    return text_chunks


# =========================================================
# HUGGINGFACE EMBEDDINGS
# =========================================================

def load_embedding():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return embeddings