# AI Model for Autonomous Driving Tasks

This project showcases the development and integration of an advanced AI model capable of performing multiple autonomous driving-related tasks. The AI model demonstrates significant improvements in accuracy and has been successfully integrated into a web application using the Django framework.

## Features

- **Lane and Line Detection**: The AI model accurately detects lanes and lines on the road, providing essential information for vehicle navigation.
- **Vehicle Identification**: Identifies and classifies different vehicles on the road, enhancing situational awareness.
- **Drivable Area Segmentation**: Segments the drivable areas from the non-drivable ones, ensuring safe navigation paths.
- **Distance Computation**: Computes the distance between the host vehicle and other objects, crucial for maintaining safe distances.

## Achievements

- **Accuracy Improvement**: The model's accuracy was significantly improved from 45% to 75.8%, ensuring more reliable performance in real-world scenarios.
- **Web Integration**: The AI model has been integrated into a web application using the Django framework, allowing for easy accessibility and interaction.

## Technologies Used

- **Python**: The core programming language used for developing the AI model.
- **TensorFlow/Keras**: Used for building and training the deep learning models.
- **OpenCV**: Utilized for image processing tasks.
- **Django**: The web framework used for integrating the AI model into a web application.
- **HTML/CSS/JavaScript**: For creating a user-friendly web interface.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/ai-autonomous-driving.git
    cd ai-autonomous-driving
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Apply migrations and run the Django server:
    ```bash
    python manage.py migrate
    python manage.py runserver
    ```

## Usage

1. Access the web application at `http://localhost:8000`.
2. Upload images or video streams to analyze lanes, vehicles, and drivable areas.
3. View the results with highlighted lanes, identified vehicles, and segmented drivable areas.

## Contributing

Contributions are welcome! Please submit a pull request or open an issue to discuss potential changes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Contact

For any inquiries or support, please contact [your-email@example.com](mailto:your-email@example.com).

---

By leveraging advanced AI techniques, this project aims to enhance the safety and reliability of autonomous driving systems. The significant improvement in accuracy and the seamless web integration highlight the potential of AI in transforming the future of transportation.
