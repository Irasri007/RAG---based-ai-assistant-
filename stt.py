import whisper

model = whisper.load_model("small")

result = model.transcribe(
    audio="audios/6 _SEO and Core Web Vitals in HTML.mp3",
    language="hi",
    task="translate"
)

print(result["text"])