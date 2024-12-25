// document.addEventListener('DOMContentLoaded', () => {
//     const history = document.getElementById('history');
//     const userQueryInput = document.getElementById('userQuery');
//     const sendQueryButton = document.getElementById('sendQuery');

//     const addToHistory = (message, isUser = true) => {
//         const messageDiv = document.createElement('div');
//         messageDiv.className = isUser ? 'user-message' : 'bot-message';
//         messageDiv.textContent = message;
//         history.appendChild(messageDiv);
//         history.scrollTop = history.scrollHeight;
//     };

//     const displayRecommendationsInHistory = (recommendations) => {
//         recommendations.forEach((rec) => {
//             const card = document.createElement('div');
//             card.className = 'card';

//             card.innerHTML = `
//                 <div class="card-container">
//                     <h3>${rec.productName}</h3>
//                     <h4>${rec.mainCatCode}</h4>
//                     <p>${rec.description}</p>
//                     <p class="price">Price: $${rec.price}</p>
//                 </div>
//             `;

//             history.appendChild(card);
//         });

//         history.scrollTop = history.scrollHeight;
//     };

//     const fetchRecommendations = async (query) => {
//         try {
//             const response = await fetch('/recommend', {
//                 method: 'POST',
//                 headers: {
//                     'Content-Type': 'application/json',
//                 },
//                 body: JSON.stringify({ query }),
//             });

//             if (!response.ok) {
//                 throw new Error('Failed to fetch recommendations');
//             }

//             const data = await response.json();
//             if (data.recommendations) {
//                 displayRecommendationsInHistory(data.recommendations);
//             } else {
//                 addToHistory('No recommendations found.', false);
//             }
//         } catch (error) {
//             addToHistory(`Error: ${error.message}`, false);
//         }
//     };

//     sendQueryButton.addEventListener('click', () => {
//         const query = userQueryInput.value.trim();
//         if (query) {
//             addToHistory(query);
//             userQueryInput.value = '';
//             fetchRecommendations(query);
//         }
//     });

//     userQueryInput.addEventListener('keypress', (event) => {
//         if (event.key === 'Enter') {
//             sendQueryButton.click();
//         }
//     });
// });


document.addEventListener('DOMContentLoaded', () => {
    const history = document.getElementById('history');
    const userQueryInput = document.getElementById('userQuery');
    const sendQueryButton = document.getElementById('sendQuery');

    const addToHistory = (message, isUser = true) => {
        const messageDiv = document.createElement('div');
        messageDiv.className = isUser ? 'user-message' : 'bot-message';
        messageDiv.textContent = message;
        history.appendChild(messageDiv);
        history.scrollTop = history.scrollHeight;
    };

    const displayRecommendationsInHistory = (recommendations) => {
        const cardRow = document.createElement('div');
        cardRow.className = 'card-row';

        recommendations.forEach((rec) => {
            const card = document.createElement('div');
            card.className = 'card';

            card.innerHTML = `
                <div class="card-container">
                    <h3>${rec.productName}</h3>
                    <h4>${rec.mainCatCode}</h4>
                    <p>${rec.description}</p>
                    <p class="price">Price: $${rec.price}</p>
                </div>
            `;

            cardRow.appendChild(card);
        });

        history.appendChild(cardRow);
        history.scrollTop = history.scrollHeight;
    };

    const fetchRecommendations = async (query) => {
        try {
            const response = await fetch('/recommend', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ query }),
            });

            if (!response.ok) {
                throw new Error('Failed to fetch recommendations');
            }

            const data = await response.json();
            if (data.recommendations) {
                displayRecommendationsInHistory(data.recommendations);
            } else {
                addToHistory('No recommendations found.', false);
            }
        } catch (error) {
            addToHistory(`Error: ${error.message}`, false);
        }
    };

    sendQueryButton.addEventListener('click', () => {
        const query = userQueryInput.value.trim();
        if (query) {
            addToHistory(query);
            userQueryInput.value = '';
            fetchRecommendations(query);
        }
    });

    userQueryInput.addEventListener('keypress', (event) => {
        if (event.key === 'Enter') {
            sendQueryButton.click();
        }
    });
});
