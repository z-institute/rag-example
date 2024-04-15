from selenium import webdriver
from bs4 import BeautifulSoup
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import time

# URL of the Notion public page you want to scrape
url = 'https://the-z.notion.site/AI-4cda91de52aa40cb91114f9b78be456b'

# Set up the Selenium WebDriver (example with Chrome)
# Make sure you have chromedriver installed and in your PATH
driver = webdriver.Chrome()

# Navigate to the page
driver.get(url)

# Wait for the page to load dynamically. Adjust the time as needed
time.sleep(5)  # Sleep for 5 seconds; adjust based on your network speed

# Get the page source and parse it with BeautifulSoup
soup = BeautifulSoup(driver.page_source, 'html.parser')

# Extract content you're interested in
# Adjust this based on your needs. For this example, let's say we extract the whole body.
content = soup.find('body')

# Specify the output HTML file path
output_html_path = 'source_docs/notion_page_content.html'

# Save the extracted HTML content to a file
with open(output_html_path, 'w', encoding='utf-8') as file:
    file.write(str(content))

# Close the WebDriver
driver.quit()

print(f'Content has been saved to {output_html_path}')