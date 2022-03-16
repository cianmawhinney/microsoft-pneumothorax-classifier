const data = new FormData();
const images = document.querySelector('input[type="file"][multiple]');
for (let i = 0; i < images.files.length; i++) {
    data.append('img_${i}', images.files[i]);
}

fetch('https://wherever.it.goes/newData', {
    method = 'POST',
    body: data,
})
.then(response => response.json())
.then(result => {
    console.log('Success:', result);
})
.catch(error => {
    console.error('Error:', error);
});