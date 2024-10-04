print('Running example job script!')
BLOB_PATH = '/blob_data'

# write a txt file to the blob storage
with open(BLOB_PATH + '/example_python_output.txt', 'w') as f:
    f.write('Hello World from example python script!')
