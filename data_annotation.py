# """
# Coding Exercise: Decoding a Secret Message
# Fetches a published Google Doc, parses the unicode character table,
# and prints the 2D grid to reveal the hidden message.
# """

# import requests
# from bs4 import BeautifulSoup


# def decode_secret_message(url: str) -> None:
#     """
#     Fetches a published Google Doc at the given URL, parses the table of
#     (x-coordinate, character, y-coordinate), builds a 2D grid, and prints it.

#     Args:
#         url: The URL of a published Google Doc containing the character table.
#     """
#     # --- 1. Fetch the document ---
#     response = requests.get(url)
#     response.raise_for_status()

#     # --- 2. Parse the HTML and find the table ---
#     soup = BeautifulSoup(response.text, "html.parser")
#     table = soup.find("table")
#     if not table:
#         raise ValueError("No table found in the document.")

#     rows = table.find_all("tr")
#     if not rows:
#         raise ValueError("Table has no rows.")

#     # --- 3. Identify column positions from the header row ---
#     headers = [th.get_text(strip=True).lower() for th in rows[0].find_all(["th", "td"])]

#     try:
#         x_idx = next(i for i, h in enumerate(headers) if "x" in h)
#         y_idx = next(i for i, h in enumerate(headers) if "y" in h)
#         char_idx = next(i for i, h in enumerate(headers) if "char" in h)
#     except StopIteration:
#         raise ValueError(f"Could not find required columns. Found headers: {headers}")

#     # --- 4. Read all data rows ---
#     data = []
#     for row in rows[1:]:
#         cells = row.find_all("td")
#         if len(cells) <= max(x_idx, y_idx, char_idx):
#             continue
#         try:
#             x = int(cells[x_idx].get_text(strip=True))
#             y = int(cells[y_idx].get_text(strip=True))
#             ch = cells[char_idx].get_text(strip=True)
#             if ch:  # skip empty character cells
#                 data.append((x, y, ch))
#         except (ValueError, IndexError):
#             continue  # skip malformed rows

#     if not data:
#         raise ValueError("No valid data rows found in the table.")

#     # --- 5. Build the grid ---
#     # x increases to the right (columns), y increases downward (rows)
#     max_x = max(x for x, _, _ in data)
#     max_y = max(y for _, y, _ in data)

#     # Initialize grid with spaces
#     grid = [[" "] * (max_x + 1) for _ in range(max_y + 1)]

#     for x, y, ch in data:
#         grid[y][x] = ch

#     # --- 6. Print the grid ---
#     print(f"Grid size: {max_x + 1} columns × {max_y + 1} rows ({len(data)} characters)\n")
#     for row in grid:
#         print("".join(row))


# if __name__ == "__main__":
#     # Example document from the coding exercise
#     DOC_URL = (
#         "https://docs.google.com/document/d/e/2PACX-1vTMOmshQe8YvaRXi6gEPKKlsC6UpFJSMAk4mQjLm_u1gmHdVVTaeh7nBNFBRlui0sTZ-snGwZM4DBCT/pub"
#     )
#     decode_secret_message(DOC_URL)


import requests
from bs4 import BeautifulSoup

def decode_secret_message(url):
    response = requests.get(url)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    rows = soup.find("table").find_all("tr")

    # figure out which column is which from the header
    headers = [td.get_text(strip=True).lower() for td in rows[0].find_all("td")]
    x_col = next(i for i, h in enumerate(headers) if "x" in h)
    y_col = next(i for i, h in enumerate(headers) if "y" in h)
    ch_col = next(i for i, h in enumerate(headers) if "char" in h)

    # read each row into (x, y, character)
    data = []
    for row in rows[1:]:
        cells = row.find_all("td")
        x = int(cells[x_col].get_text(strip=True))
        y = int(cells[y_col].get_text(strip=True))
        ch = cells[ch_col].get_text(strip=True)
        data.append((x, y, ch))

    # build the grid: x = column, y = row with 0 at the bottom
    max_x = max(x for x, _, _ in data)
    max_y = max(y for _, y, _ in data)

    grid = [[" "] * (max_x + 1) for _ in range(max_y + 1)]
    for x, y, ch in data:
        grid[y][x] = ch

    # y=0 is the bottom row, so print in reverse
    for row in reversed(grid):
        print("".join(row))


URL = "https://docs.google.com/document/d/e/2PACX-1vTMOmshQe8YvaRXi6gEPKKlsC6UpFJSMAk4mQjLm_u1gmHdVVTaeh7nBNFBRlui0sTZ-snGwZM4DBCT/pub"
decode_secret_message(URL)