def read_file(file_path='../NBA.txt'):
    # Read the file
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return content
    except FileNotFoundError as e:
        print('File not found:', e)
    except PermissionError as e:
        print('Permission denied:', e)
    except Exception as e:
        print('Error occurred:', e)
        return None
