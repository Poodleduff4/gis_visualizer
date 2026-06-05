# serve.py
from RangeHTTPServer import RangeRequestHandler
import http.server


class CORSHandler(RangeRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory="assets", **kwargs)

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Range")
        self.send_header(
            "Access-Control-Expose-Headers", "Content-Range, Accept-Ranges"
        )
        self.send_header("Cache-Control", "public, max-age=3600")
        super().end_headers()


http.server.ThreadingHTTPServer(("", 8001), CORSHandler).serve_forever()
