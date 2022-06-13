'use strict';

function imagePreviewHandler(event) {
    if (event.target.files && event.target.files[0]) {
        let reader = new FileReader();
        reader.onload = function (e) {
            let img = document.querySelector('.background-preview > img');
            img.src = e.target.result;
            if (img.classList.contains('d-none')) {
                let label = document.querySelector('.background-preview > label');
                label.classList.add('d-none');
                img.classList.remove('d-none');
            }
        }
        reader.readAsDataURL(event.target.files[0]);
    }
}

function openLink(event) {
    let row = event.target.closest('.row');
    if (row.dataset.url) {
        window.location = row.dataset.url;
    }
}

function imageUploadFunction(file, onSuccess, onError) {
    let xhr = new XMLHttpRequest();
    let formData = new FormData();
    let formElm = this.element.closest('form');
    xhr.responseType = 'json';
    xhr.open('POST', '/api/images/upload');
    xhr.onload = function() {
        if (this.status == 200) {
            onSuccess(this.response.data.filePath);

            let hiddenField = document.createElement('input');
            hiddenField.type = 'hidden';
            hiddenField.name = 'image_id';
            hiddenField.value = this.response.data.imageId;
            formElm.append(hiddenField);

        } else {
            onError(this.response.error);
        }
    };
    formData.append("image", file);
    xhr.send(formData);
}

const TOOLBAR_ITEMS = [
    "bold", "italic", "heading", "|", 
    "quote", "ordered-list", "unordered-list", "|",
    "link", "upload-image", "|",  
    "preview", "side-by-side", "fullscreen", "|",
    "guide"
]

window.onload = function() {
    let background_img_field = document.getElementById('background_img');
    if (background_img_field) {
        background_img_field.onchange = imagePreviewHandler;
    }
    for (let course_elm of document.querySelectorAll('.courses-list .row')) {
        course_elm.onclick = openLink;
    }
    if (document.getElementById('text-content')) {
        let easyMDE = new EasyMDE({
            element: document.getElementById('text-content'),
            toolbar: TOOLBAR_ITEMS,
            uploadImage: true,
            imageUploadEndpoint: '/api/images/upload',
            imageUploadFunction: imageUploadFunction
        });
    }
}