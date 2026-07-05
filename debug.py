import re

with open("WhatsApp Chat with Davv friends.txt", "rb") as f:
    raw = f.read()

try:
    data = raw.decode("utf-8")
except UnicodeDecodeError:
    data = raw.decode("utf-16")

pattern = r'\n\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s-\s'
dates = re.findall(pattern, data)

print("Total dates matched:", len(dates))
print("First 5 matches:", dates[:5])
print("First 300 characters of file (repr):")
print(repr(data[:300]))