import json
import os

with open("locations.JSON") as data_file:
    data = json.load(data_file)

exceptions = {
    "China-Shanghai": 
    {
        "url": "https://www.countries-ofthe-world.com/flags/flag-of-China.png",
        "replace-name": "China-Shanghai"
    },
    "Chinese Taipei": 
    {
        "url": "https://www.countries-ofthe-world.com/flags/flag-of-Taiwan.png",
        "replace-name": "Chinese-Taipei"
    },
    "Hong Kong-China":
    {
        "url":  "https://www.countries-ofthe-world.com/flags/flag-of-China.png",
        "replace-name": "Hong-Kong-China"
    },
    "Korea": 
    {
        "url": "https://www.countries-ofthe-world.com/flags/flag-of-Korea-South.png",
        "replace-name": "Korea"
    },
    "Macao-China": 
    {
        "url": "https://www.countries-ofthe-world.com/flags/flag-of-China.png",
        "replace-name": "Macao-China"
    },
    "Perm(Russian Federation)": 
    {
        "url": "https://www.countries-ofthe-world.com/flags/flag-of-Russia.png",
        "replace-name": "Perm-Russian-Federation"
    },
    "Russian Federation": 
    {
        "url": "https://www.countries-ofthe-world.com/flags/flag-of-Russia.png",
        "replace-name": "Russian-Federation"
    },
    "Slovak Republic": 
    {
        "url": "https://www.countries-ofthe-world.com/flags/flag-of-Slovakia.png",
        "replace-name": "Slovak-Republic"
    },
};

for item in data:
    the_country = item["country"]
    if the_country.replace(" ", "-") not in exceptions.keys():
        modded = the_country.replace(" ", "-")
        os.system("curl https://www.countries-ofthe-world.com/flags/flag-of-{0}.png > flags/flag-of-{0}.png".format(modded))

    else:
        os.system("curl {0} > flags/flag-of-{1}.png".format(exceptions[the_country]["url"],
                    exceptions[the_country]["replace-name"]
                  ))

    print the_country
