import requests
import time

API_URL = "https://elonmuskarchive.org/agents/index"
OUTPUT_FILE = "elon_tweets.txt"

LIMIT = 1000


def get_page(offset=0):
    """Get one page from the Elon Musk Archive API."""

    params = {
        "type": "posts",
        "list": "1",
        "limit": LIMIT,
        "offset": offset,
        "fields": "id,date,title,url",
        "sort": "date_asc",
    }

    r = requests.get(
        API_URL,
        params=params,
        timeout=60
    )

    r.raise_for_status()

    data = r.json()

    # Uncomment this if you want to see the raw API response
    # print(data)

    return data


def main():

    print("=" * 60)
    print("ELON MUSK ARCHIVE DOWNLOADER")
    print("=" * 60)
    print()

    # --------------------------------------------------------
    # First request
    # --------------------------------------------------------

    print("Connecting to archive...")

    data = get_page(0)

    total = data.get("total", 0)
    entries = data.get("entries", [])

    print(f"Archive reports: {total:,} posts")
    print(f"First request returned: {len(entries):,}")
    print()

    if not entries:
        print("ERROR: API returned zero entries.")
        print()
        print("Raw response:")
        print(data)
        return

    # --------------------------------------------------------
    # Download
    # --------------------------------------------------------

    all_entries = entries

    offset = len(entries)

    while offset < total:

        print(
            f"Downloading {offset + 1:,} - "
            f"{min(offset + LIMIT, total):,} "
            f"of {total:,}"
        )

        data = get_page(offset)

        entries = data.get("entries", [])

        if not entries:
            print("API returned no more entries.")
            break

        all_entries.extend(entries)

        offset += len(entries)

        print(
            f"  Received {len(entries):,} "
            f"(total downloaded: {len(all_entries):,})"
        )

        time.sleep(0.05)

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    print()
    print("Writing file...")

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        for post in all_entries:

            date = post.get("date", "")
            text = post.get("title", "")

            # API gives YYYY-MM-DD
            # Convert to MM/DD/YYYY

            try:
                year, month, day = date.split("-")
                formatted_date = f"{month}/{day}/{year}"

            except Exception:
                formatted_date = date

            # Keep one post per line
            text = str(text)
            text = text.replace("\r", " ")
            text = text.replace("\n", " ")

            f.write(
                f"{formatted_date} : {text}\n"
            )

    print()
    print("=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"Posts downloaded : {len(all_entries):,}")
    print(f"Output file      : {OUTPUT_FILE}")
    print()


if __name__ == "__main__":
    main()