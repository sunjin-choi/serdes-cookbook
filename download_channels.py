import os
import urllib.request

def download_if_missing(url: str, filename: str, directory=".") -> None:
    """Download file from URL if it does not already exist."""
    # if the directory does not exist, create it
    if not os.path.exists(directory):
        os.makedirs(directory)

    if os.path.exists(filename):
        print(f"File already exists: {filename}")
        return

    print(f"Downloading {filename} from {url}...")
    try:
        urllib.request.urlretrieve(url, filename)
        # move file to the specified directory
        if directory != ".":
            os.rename(filename, os.path.join(directory, filename))
        print(f"Downloaded successfully: {filename}")
    except Exception as e:
        print(f"Failed to download {url}: {e}")

if __name__ == "__main__":
    # Replace with your actual URL and desired local filename
    FILE_URL = "https://www.ieee802.org/3/ck/public/tools/cucable/mellitz_3ck_04_1119_CACR.zip"
    LOCAL_FILENAME = "mellitz_3ck_04_1119_CACR.zip"

    download_if_missing(FILE_URL, LOCAL_FILENAME, directory="example_channel")
