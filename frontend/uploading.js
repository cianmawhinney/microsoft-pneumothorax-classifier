async function uploadImage() {
  const endpointURL = "https://pneumothorax.mawh.in/score"
  const apiKey = "LDzXDtJKHklAX2uhlDHRdbf5DRcTgXYf";

  const data = new FormData();
  const images = document.getElementById('customFile');
  // const images = document.querySelector('input[type="file"][multiple]');
  for (let i = 0; i < images.files.length; i++) {
    data.append('image', images.files[i]);
  }
  
  await fetch(endpointURL, {
    method: 'POST',
    body: data,
    headers: {
      'Authorization': `Bearer ${apiKey}`,
    },
  })
  .then(response => response.json())
  .then(result => {
    console.log('Success:', result);
  })
  .catch(error => {
    console.error('Error:', error);
  });
}
