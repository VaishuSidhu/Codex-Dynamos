// scripts.js

document.addEventListener("DOMContentLoaded", function () {
    // Your JavaScript code goes here
    var additionalBoxCount = 1;

    // Function to add another box
    function addAnotherBox() {
        var newBox = document.createElement("div");
        newBox.innerHTML = `
            <p id="additional-question-text">Enter your questions:</p>
            <textarea id="additional-answer-input-${additionalBoxCount}" placeholder="Type your answer here"></textarea>
        `;
        document.getElementById("feedback-box").appendChild(newBox);
        additionalBoxCount++;
    }

    // Event listener for the "Add Another Box" button
    document.getElementById("add-box").addEventListener("click", addAnotherBox);

    // Event listener for the "Get Link" button
    document.getElementById("get-link").addEventListener("click", function () {
        // Your code to generate a link based on user input goes here
        alert("Link generated!");
    });
});
