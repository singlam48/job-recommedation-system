from django.shortcuts import render
from django.http import HttpResponse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from django.conf import settings
from django.core.files.storage import FileSystemStorage
import docx2txt 
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Create your views here.

def home(request):
    return render(request, 'index.html', {"name":"ranjeet"})


def upload(request):
    return render(request, 'main.html', {"name":"ranjeet"})

def resumeScanner(request):
    
    # Getting Data from frontend
  
    job_description = request.POST["jd"]
    document = request.FILES['resume']

    # Saving the uploaded documents
    fs = FileSystemStorage()
    filename = fs.save(document.name, document)
    uploaded_file_url = fs.url(filename)

    try:
        resume = docx2txt.process(os.path.join(BASE_DIR/'media', filename))

            # print(resume)
            # print(os.path.join(BASE_DIR/'media', filename))
            # print(BASE_DIR)
        
        
            # Processing the data
        content_list = [job_description,resume]
        cv = CountVectorizer()
        count_matrix = cv.fit_transform(content_list)
        
        matrix = cosine_similarity(count_matrix)
        result = (matrix[1][0]*100).round(2)

        fs.delete(filename)

        return render(request, 'main.html', {"result":result})
        
    except:

        print("something went wrong")
        return HttpResponse("please upload docs File ........")
 

 
    
    
