from django.shortcuts import render
from .ai_models import *
from PIL import Image  # For image processing
from .ai_models import ImageAIDetector
import io

text_detector = TextAIDetector()
image_detector = ImageAIDetector()
audio_detector = AudioAIDetector()

# Create your views here.
def index(request):
    if request.method == "GET":
        return render(request, "index.html")
   
            
        

def audio(request):
    if request.method == "GET":
        return render(request, "audio.html")
    elif request.method == "POST":
        audio_file = request.FILES.get("audio",None)
        if audio_file:
            # Use io.BytesIO to pass the file to the model for reading
            wav_file = io.BytesIO(audio_file.read())

            # Get the prediction from the audio detector
            prediction = audio_detector.predict(wav_file)
            return render(request,"audio.html",{"prediction": prediction})
        else:
            pass
        


def words(request):
    if request.method == "GET":
        return render(request, "words.html")
    elif request.method == "POST":
            user_text = request.POST.get("text",None)
            if user_text:
                prediction = text_detector.predict(user_text)
                return render(request, "words.html", {"prediction": prediction})

def pics(request):
    if request.method == "GET":
        return render(request, "pics.html")
    elif request.method == "POST":
        user_image = request.FILES.get("image", None)
        # uploaded_file = request.FILES['image']

        # Open the file as a PIL image
        image = Image.open(user_image)

            # Ensure the image is in RGB mode (some models may require it)
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        prediction = image_detector.predict(image)
        return render(request, "pics.html", {"prediction": prediction})