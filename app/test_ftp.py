#%%
import ftplib
#%%
# Define the FTP server information
FTP_HOST = "ftp.meraki-it.pk"
FTP_USER = "MassAI@front.meraki-it.pk"
FTP_PASS = "MassAI@2023"

#%%
remote_path = "/"
public_url = "front.meraki-it.pk/MassAI/"
# remote_path = "front.meraki-it.pk/MassAI/"
# Connect to the FTP server
with ftplib.FTP(FTP_HOST, FTP_USER, FTP_PASS) as ftp:
    # Change the current directory if needed
    ftp.cwd(remote_path)

    # Open the local file in binary mode
    with open('localfile.txt', 'rb') as file:
        # Upload the file to the server
        ftp.storbinary('STOR remote_file.txt', file)

    # Get the upload URL
    upload_url = f"ftp://{FTP_USER}@{FTP_HOST}"+remote_path+"remote_file.txt"
    print("File uploaded successfully. URL:", upload_url)
# %%
