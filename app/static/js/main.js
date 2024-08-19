document.querySelector('form').addEventListener('submit', async (e) => {
    e.preventDefault();

    const formData = new FormData(e.target);

    const response = await fetch('/import', {
        method: 'POST',
        body: formData,
    });

    if (response.ok) {
        const html = await response.text();
        document.querySelector('.results').innerHTML = html;
    } else {
        alert('Failed to process files. Please try again.');
    }
});
