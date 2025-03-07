# Numerical Syntax Parser and Analyzer

A web application for analyzing English numeratives based on Brett Reynolds' framework presented in "Numerical Syntax: Toward a Proper Analysis of English Numeratives".

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)

## Overview

This tool provides linguistic analysis of English numerical expressions by implementing Reynolds' hybrid approach where basic numeratives are treated as lexemes, and larger expressions as syntactic constructions.

![Syntax Tree Example](https://github.com/username/numerical-syntax-parser/raw/main/example_tree.png)

## Features

- **Parse and Visualize**: Transform numerical expressions into their syntactic structure and view tree visualizations
- **Number Conversion**: Convert between numerical and verbal representations 
- **Lexical Analysis**: Identify the syntactic categories of numeratives (determinative, proper noun, common noun, adjective)
- **Validation**: Flag common errors in numerical expressions according to linguistic principles

## Try It Online

The application is deployed on Streamlit Cloud and can be accessed here: [Numerical Syntax Parser App](https://share.streamlit.io/username/numerical-syntax-parser/streamlit_app.py)

## How to Use

### Parse and Visualize
Enter a numerative expression (e.g., "two thousand and seven") to see its syntactic structure displayed as text and as a tree visualization.

### Convert Numbers to Words
Enter any number to convert it to its English verbal form.

### Convert Words to Numbers
Enter any English numerative expression to see its numerical value.

### Determine Lexical Category
Enter a numerative expression with context to determine its lexical category (e.g., "three books", "the seventh day").

### Validate Expression
Check if a numerative expression follows the proper syntax rules according to Reynolds' framework.

## Local Installation

To run this application locally:

1. Clone this repository:
```bash
git clone https://github.com/username/numerical-syntax-parser.git
cd numerical-syntax-parser
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Run the Streamlit app:
```bash
streamlit run streamlit_app.py
```

## Requirements

- Python 3.7+
- Streamlit
- NLTK
- Matplotlib
- NumPy
- Pillow

## Theoretical Framework

This application implements the framework developed by Brett Reynolds in "Numerical Syntax: Toward a Proper Analysis of English Numeratives." The approach treats basic numeratives (0-99) as lexemes and larger expressions as syntactic constructions, providing a hybrid model for analyzing numerical expressions.

Key concepts include:
- The distinction between determinative, proper noun, common noun, and adjective uses of numeratives
- The structural analysis of complex numeratives using coordination ("and") and factor-magnitude patterns
- The validation of numerative expressions according to syntactic rules

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Brett Reynolds for the theoretical framework on numerical syntax
- NLTK for providing the natural language processing tools
- Streamlit for the web application framework
