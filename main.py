import logging


# Set up logging
logging.basicConfig(filename='main.log', level=logging.INFO)






def main():
    print('start')
    logging.info(f"Starting addition")

    try:
        denominator = 's'  # Example denominator
        if denominator == 0:
            raise ValueError("Denominator cannot be zero. =)")
        x = 10 + denominator
    except Exception as e:
        logging.error(f"Error: {e}")
        print(f"Error: {e}")  # This will also propagate the error










if __name__ == "__main__":
    
    main()  # Call the main function