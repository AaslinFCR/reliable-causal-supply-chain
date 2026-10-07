"""Read public portal page metadata without executing page code."""

import json
from html.parser import HTMLParser
from pathlib import Path


class Page(HTMLParser):
    def __init__(self):
        super().__init__()
        self.scripts = []
        self.collect = False
        self.data = []

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if tag == "script":
            if "src" in values:
                self.scripts.append(values["src"])
            self.collect = values.get("id") == "__NEXT_DATA__"

    def handle_data(self, data):
        if self.collect:
            self.data.append(data)

    def handle_endtag(self, tag):
        if tag == "script":
            self.collect = False


page = Page()
page.feed(Path("reports/idp_portal.html").read_text(encoding="utf-8"))
print(json.dumps(page.scripts))
raw = "".join(page.data) or "{}"
Path("reports/idp_page_data.json").write_text(raw, encoding="utf-8")
print(raw[:18000])
