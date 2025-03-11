import os
import requests

def download_doodle(doodle_name):
    url = f'https://storage.googleapis.com/quickdraw_dataset/full/simplified/{doodle_name}.ndjson'
    response = requests.get(url)

    if response.status_code == 200:
        return response.text
    else:
        print(f"Failed to download {doodle_name}: {response.status_code}")
        return None

def save_doodle(doodle_name, data):
    if not os.path.exists('data'):
        os.makedirs('data')

    file_path = os.path.join('data', f'{doodle_name}.ndjson')
    with open(file_path, 'w') as file:
        file.write(data)
    print(f"Saved {doodle_name} to {file_path}")

def main():
    try:
        with open('doodles_to_download.txt', 'r') as file:
            doodles = file.read().splitlines()

        for doodle in doodles:
            print(f"Downloading {doodle}...")
            doodle_data = download_doodle(doodle)
            if doodle_data:
                save_doodle(doodle, doodle_data)

    except FileNotFoundError:
        print("The file 'doodles_to_download.txt' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    main()
