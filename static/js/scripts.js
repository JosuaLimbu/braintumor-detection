function previewImage(event) {
    const fileInput = event.target;
    const preview = document.getElementById("image-preview");
    const placeholder = document.querySelector(".placeholder");
    const fileName = document.getElementById("file-name");

    if (fileInput.files && fileInput.files[0]) {
        const reader = new FileReader();

        reader.onload = function (e) {
            preview.src = e.target.result;
            preview.style.display = "block";
            placeholder.style.display = "none"; 
        };

        reader.readAsDataURL(fileInput.files[0]);

        fileName.textContent = `Selected file: ${fileInput.files[0].name}`;
    } else {
        preview.style.display = "none";
        placeholder.style.display = "block";
        fileName.textContent = "No file selected";
    }
}
