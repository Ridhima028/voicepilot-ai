FROM python:3.11-slim

WORKDIR /app

# ffmpeg: yt-dlp needs it to extract/convert YouTube audio to .m4a.
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*

# deno: YouTube now serves a JavaScript challenge, and yt-dlp solves it with
# an external JS runtime. Without one, extraction is deprecated and formats
# go missing. Copied as a single static binary from the official image.
COPY --from=denoland/deno:bin-2.5.6 /deno /usr/local/bin/deno

COPY requirements.txt .

# No PyTorch dependency anymore — semantic segmentation now uses the Gemini
# embeddings API instead of a local sentence-transformers model, so this
# build no longer needs the special CPU-torch install step that used to
# live here. Builds should be noticeably faster now.
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]