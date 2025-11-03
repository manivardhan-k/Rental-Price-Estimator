document.addEventListener("DOMContentLoaded", () => {
    // Select all text inputs inside your form/section
    const inputs = document.querySelectorAll(".input-group input[type='text']");
  
    inputs.forEach(input => {
      const placeholderLength = input.placeholder.length;
      input.style.width = `${placeholderLength + 1}ch`; // +2ch for padding
    });
  });
  