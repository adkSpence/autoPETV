"""
Minimal seekable, read-only file-like object over an HTTP resource that
supports Range requests -- lets zipfile.ZipFile read a remote zip's
central directory and individual entries without downloading the whole
archive.

Re-issues a fresh request (following redirects) for every read, since the
FDAT repository's download link redirects to a presigned S3 URL that
expires after 60 seconds -- a single cached signed URL isn't safe to
reuse across a long-running zip listing/extraction.
"""
import requests


class HttpRangeFile:
    def __init__(self, url: str):
        self.url = url
        self.session = requests.Session()
        head = self.session.head(url, allow_redirects=True)
        # The redirect target (presigned S3 URL) rejects HEAD (signed for
        # GET only), so get size via a 1-byte ranged GET instead.
        r = self.session.get(url, headers={"Range": "bytes=0-0"})
        r.raise_for_status()
        content_range = r.headers["Content-Range"]  # "bytes 0-0/<total>"
        self._size = int(content_range.split("/")[-1])
        self._pos = 0

    def seekable(self):
        return True

    def seek(self, offset, whence=0):
        if whence == 0:
            self._pos = offset
        elif whence == 1:
            self._pos += offset
        elif whence == 2:
            self._pos = self._size + offset
        return self._pos

    def tell(self):
        return self._pos

    def read(self, size=-1):
        if size is None or size < 0:
            end = self._size - 1
        else:
            end = min(self._pos + size, self._size) - 1
        if self._pos > end:
            return b""
        r = self.session.get(self.url, headers={"Range": f"bytes={self._pos}-{end}"})
        r.raise_for_status()
        data = r.content
        self._pos += len(data)
        return data

    def __len__(self):
        return self._size
