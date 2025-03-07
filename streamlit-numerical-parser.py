"""
Numerical Syntax Parser and Analyzer - Streamlit Web Application

This program implements a parser and analyzer for English numeratives based on the 
framework developed in "Numerical Syntax: Toward a Proper Analysis of English Numeratives"
by Brett Reynolds.
"""

import streamlit as st
import re
import nltk
from nltk import Tree
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import io
from PIL import Image

# Ensure NLTK components are available (uncomment for first run)
# nltk.download('punkt')

class NumericalSyntaxParser:
    """Parser for English numeratives following Reynolds' framework"""
    
    def __init__(self):
        # Basic numeratives (0-99)
        self.single_digits = {
            'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 
            'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9
        }
        
        self.teens = {
            'ten': 10, 'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14,
            'fifteen': 15, 'sixteen': 16, 'seventeen': 17, 'eighteen': 18, 'nineteen': 19
        }
        
        self.decades = {
            'twenty': 20, 'thirty': 30, 'forty': 40, 'fifty': 50,
            'sixty': 60, 'seventy': 70, 'eighty': 80, 'ninety': 90
        }
        
        # Magnitude words
        self.magnitudes = {
            'hundred': 100,
            'thousand': 1000,
            'million': 1000000,
            'billion': 1000000000,
            'trillion': 1000000000000
        }
        
        # Ordinal suffixes and irregular forms
        self.ordinal_suffixes = {
            'one': 'first', 'two': 'second', 'three': 'third', 'five': 'fifth',
            'eight': 'eighth', 'nine': 'ninth', 'twelve': 'twelfth'
        }
        
        # For generation and conversion
        self.num_to_word = {}
        for d in [self.single_digits, self.teens, self.decades, self.magnitudes]:
            self.num_to_word.update({v: k for k, v in d.items()})
    
    def is_basic_numerative(self, word):
        """Check if a word is a basic numerative (0-99)"""
        return (word in self.single_digits or 
                word in self.teens or 
                word in self.decades or
                word.startswith(tuple(self.decades.keys())) and 
                '-' in word and 
                word.split('-')[1] in self.single_digits)
    
    def is_magnitude(self, word):
        """Check if a word is a magnitude term"""
        return word in self.magnitudes
    
    def tokenize_numerative(self, text):
        """Split a verbal numerative into tokens"""
        # Handle hyphenated compound numeratives
        tokens = []
        for part in text.lower().split():
            if '-' in part and not part.endswith('th'):
                tokens.extend(part.split('-'))
            else:
                tokens.append(part)
        return tokens
    
    def parse_numerative(self, text):
        """
        Parse an English numerative expression and return its syntactic structure
        as a tree, following Reynolds' hybrid approach where basic numeratives are
        lexemes and larger expressions are syntactic constructions.
        """
        tokens = self.tokenize_numerative(text)
        
        # Check if this is a basic numerative (0-99)
        if len(tokens) == 1 and self.is_basic_numerative(tokens[0]):
            return Tree('D', [tokens[0]])
        
        # Handle more complex numeratives
        return self._build_tree(tokens)
    
    def _build_tree(self, tokens):
        """Build a syntax tree for a numerative expression"""
        # Simple case: single token
        if len(tokens) == 1:
            if self.is_basic_numerative(tokens[0]):
                return Tree('D', [tokens[0]])
            elif self.is_magnitude(tokens[0]):
                return Tree('D', [tokens[0]])
            else:
                # Could be an ordinal or other form
                return Tree('Adj', [tokens[0]]) if tokens[0].endswith('th') else Tree('D', [tokens[0]])
        
        # Look for coordination with "and"
        if 'and' in tokens:
            and_index = tokens.index('and')
            left = self._build_tree(tokens[:and_index])
            right = self._build_tree(tokens[and_index+1:])
            return Tree('Coordination', [left, Tree('Crd', ['and']), right])
        
        # Look for factor + magnitude pattern
        for i, token in enumerate(tokens):
            if self.is_magnitude(token) and i > 0:
                factor = self._build_tree(tokens[:i])
                rest = tokens[i+1:] if i+1 < len(tokens) else []
                
                if not rest:  # Just factor + magnitude
                    return Tree('DP', [
                        Tree('Mod_fact', [factor]),
                        Tree('D', [token])
                    ])
                else:  # More complex structure with addition
                    magnitude = Tree('DP', [
                        Tree('Mod_fact', [factor]),
                        Tree('D', [token])
                    ])
                    
                    addition = self._build_tree(rest)
                    return Tree('Coordination', [magnitude, addition])
        
        # Fall back for other patterns
        return Tree('DP', tokens)
    
    def determine_category(self, text):
        """
        Determine the lexical category of the numerative:
        - determinative
        - proper noun
        - common noun
        - adjective
        """
        text = text.lower()
        
        # Check for ordinals (adjectives)
        if (text.endswith('th') or text.endswith('st') or 
            text.endswith('nd') or text.endswith('rd') or
            text in ['first', 'second', 'third']):
            return "Adjective (Ordinal)"
        
        # Check for fractions (common nouns)
        if ('half' in text or 'third' in text or 'quarter' in text or
            'fifth' in text or text.endswith('ths')):
            return "Common Noun (Fractional)"
        
        # Check context clues for proper vs. common noun vs. determinative
        if "is prime" in text or "equals" in text or "room" in text:
            return "Proper Noun (Cardinal)"
        elif text.endswith('s') or "of" in text:
            return "Common Noun (Cardinal)"
        else:
            return "Determinative (Cardinal)"
    
    def convert_to_number(self, text):
        """Convert a verbal numerative to its numerical value"""
        tokens = self.tokenize_numerative(text.lower().replace('-', ' ').replace('and', '').split())
        result = 0
        current_sum = 0
        
        for token in tokens:
            if token in self.single_digits:
                current_sum += self.single_digits[token]
            elif token in self.teens:
                current_sum += self.teens[token]
            elif token in self.decades:
                current_sum += self.decades[token]
            elif token in self.magnitudes:
                # Apply magnitude to current sum
                if current_sum == 0:
                    current_sum = 1
                current_sum *= self.magnitudes[token]
                result += current_sum
                current_sum = 0
            # Handle compound forms like "twenty-one"
            elif token.count('-') == 1:
                parts = token.split('-')
                if parts[0] in self.decades and parts[1] in self.single_digits:
                    current_sum += self.decades[parts[0]] + self.single_digits[parts[1]]
        
        # Add any remaining sum
        result += current_sum
        return result
    
    def convert_to_words(self, number):
        """Convert a number to its English verbal form"""
        if number == 0:
            return "zero"
        
        if number < 0:
            return "minus " + self.convert_to_words(abs(number))
        
        parts = []
        
        # Handle billions
        if number >= 1000000000:
            billions = number // 1000000000
            parts.append(self.convert_to_words(billions) + " billion")
            number %= 1000000000
        
        # Handle millions
        if number >= 1000000:
            millions = number // 1000000
            parts.append(self.convert_to_words(millions) + " million")
            number %= 1000000
        
        # Handle thousands
        if number >= 1000:
            thousands = number // 1000
            parts.append(self.convert_to_words(thousands) + " thousand")
            number %= 1000
        
        # Handle hundreds
        if number >= 100:
            hundreds = number // 100
            parts.append(self.convert_to_words(hundreds) + " hundred")
            number %= 100
        
        # Handle tens and ones
        if number > 0:
            if parts:  # Add "and" if there are already parts
                parts.append("and")
            
            if number < 20:
                for word, value in {**self.single_digits, **self.teens}.items():
                    if value == number:
                        parts.append(word)
                        break
            else:
                tens = (number // 10) * 10
                ones = number % 10
                
                tens_word = self.num_to_word[tens]
                if ones == 0:
                    parts.append(tens_word)
                else:
                    ones_word = self.num_to_word[ones]
                    parts.append(f"{tens_word}-{ones_word}")
        
        return " ".join(parts)
    
    def visualize_tree(self, tree):
        """Visualize the syntax tree of a numerative and return the image"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        def draw_tree(tree, depth=0, x_pos=0, parent_x=None, parent_y=None):
            # Calculate position
            y_pos = -depth
            
            # Draw node
            node_text = tree.label() if isinstance(tree, Tree) else str(tree)
            ax.text(x_pos, y_pos, node_text, 
                    ha='center', va='center', 
                    bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))
            
            # Draw line to parent
            if parent_x is not None and parent_y is not None:
                ax.plot([parent_x, x_pos], [parent_y, y_pos], 'k-')
            
            # Process children
            if isinstance(tree, Tree):
                num_children = len(tree)
                child_width = 2.0 / (2**depth)
                
                child_positions = []
                for i, child in enumerate(tree):
                    # Calculate position for child
                    child_x = x_pos + (i - (num_children-1)/2) * child_width * 2
                    child_positions.append(child_x)
                    
                    # Recursively draw child
                    draw_tree(child, depth+1, child_x, x_pos, y_pos)
        
        # Start drawing from the root
        draw_tree(tree)
        
        # Adjust plot limits
        ax.set_xlim(-5, 5)
        ax.set_ylim(-8, 1)
        ax.axis('off')
        ax.set_title(f"Syntax Tree: {' '.join(tree.leaves())}")
        
        # Convert plot to image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        plt.close(fig)
        return Image.open(buf)

    def validate_numerative(self, text):
        """
        Check if a numerative expression is valid according to 
        Reynolds' framework and highlight any errors.
        """
        tokens = self.tokenize_numerative(text.lower())
        issues = []
        
        # Check for valid basic numeratives
        for i, token in enumerate(tokens):
            if (token not in self.single_digits and 
                token not in self.teens and 
                token not in self.decades and 
                token not in self.magnitudes and
                token != 'and'):
                
                # Check compound forms
                if '-' in token:
                    parts = token.split('-')
                    if (len(parts) != 2 or 
                        parts[0] not in self.decades or 
                        parts[1] not in self.single_digits):
                        issues.append(f"Invalid compound numerative: '{token}'")
                else:
                    issues.append(f"Unknown numerative term: '{token}'")
        
        # Check for factor + magnitude patterns
        for i, token in enumerate(tokens):
            if self.is_magnitude(token) and i == 0:
                issues.append(f"Magnitude term '{token}' requires a factor")
                
        # Check for proper use of 'and'
        if 'and' in tokens:
            and_index = tokens.index('and')
            if and_index == 0 or and_index == len(tokens) - 1:
                issues.append("'and' cannot appear at the beginning or end")
            
            before_and = tokens[and_index - 1] if and_index > 0 else None
            after_and = tokens[and_index + 1] if and_index < len(tokens) - 1 else None
            
            if before_and and self.is_magnitude(before_and) and after_and and self.is_magnitude(after_and):
                issues.append("'and' cannot connect two magnitude terms directly")
        
        return issues if issues else ["Valid numerative expression"]


# Streamlit web app
def main():
    st.set_page_config(
        page_title="Numerical Syntax Analyzer",
        page_icon="🔢",
        layout="wide"
    )
    
    st.title("Numerical Syntax Analyzer")
    st.markdown("Based on 'Numerical Syntax' by Brett Reynolds")
    st.markdown("---")
    
    parser = NumericalSyntaxParser()
    
    # Create tabs for different functionalities
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Parse & Visualize", 
        "Number to Words", 
        "Words to Number", 
        "Lexical Category",
        "Validate Expression"
    ])
    
    # Tab 1: Parse and Visualize
    with tab1:
        st.header("Parse and Visualize Numerative Structure")
        expression = st.text_input("Enter a numerative expression:", value="two thousand and seven", key="parse_input")
        
        if st.button("Parse and Visualize", key="parse_button"):
            with st.spinner("Parsing and generating visualization..."):
                try:
                    tree = parser.parse_numerative(expression)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Syntactic Structure (Text)")
                        st.text(str(tree))
                    
                    with col2:
                        st.subheader("Syntactic Structure (Tree)")
                        img = parser.visualize_tree(tree)
                        st.image(img, use_column_width=True)
                except Exception as e:
                    st.error(f"Error processing expression: {str(e)}")
    
    # Tab 2: Number to Words
    with tab2:
        st.header("Convert Number to Words")
        number = st.number_input("Enter a number:", value=2023, min_value=-999999999999, max_value=999999999999, step=1, key="number_input")
        
        if st.button("Convert to Words", key="words_button"):
            try:
                words = parser.convert_to_words(int(number))
                st.success(f"**In words:** {words}")
            except Exception as e:
                st.error(f"Error converting number: {str(e)}")
    
    # Tab 3: Words to Number
    with tab3:
        st.header("Convert Words to Number")
        word_expression = st.text_input("Enter a numerative expression:", value="two million twenty-three", key="word_input")
        
        if st.button("Convert to Number", key="number_button"):
            try:
                number = parser.convert_to_number(word_expression)
                st.success(f"**Numerical value:** {number:,}")
            except Exception as e:
                st.error(f"Error converting expression: {str(e)}")
    
    # Tab 4: Lexical Category
    with tab4:
        st.header("Determine Lexical Category")
        category_expression = st.text_input("Enter a numerative expression with context:", 
                                           value="three books", 
                                           help="Include context like 'three books' or 'the seventh day'",
                                           key="category_input")
        
        if st.button("Determine Category", key="category_button"):
            try:
                category = parser.determine_category(category_expression)
                st.info(f"**Lexical category:** {category}")
                
                # Provide explanation based on category
                if "Adjective" in category:
                    st.markdown("**Explanation:** The numerative is functioning as an adjective, typically modifying a noun.")
                elif "Proper Noun" in category:
                    st.markdown("**Explanation:** The numerative is functioning as a proper noun, referring to a specific entity.")
                elif "Common Noun" in category:
                    st.markdown("**Explanation:** The numerative is functioning as a common noun, referring to a class of entities.")
                elif "Determinative" in category:
                    st.markdown("**Explanation:** The numerative is functioning as a determinative, specifying the reference of a noun phrase.")
            except Exception as e:
                st.error(f"Error determining category: {str(e)}")
    
    # Tab 5: Validate Expression
    with tab5:
        st.header("Validate Numerative Expression")
        validate_expression = st.text_input("Enter a numerative expression to validate:", value="twenty-five thousand", key="validate_input")
        
        if st.button("Validate", key="validate_button"):
            try:
                issues = parser.validate_numerative(validate_expression)
                
                if issues == ["Valid numerative expression"]:
                    st.success("✅ Valid numerative expression")
                else:
                    st.error("⚠️ Issues found in the expression:")
                    for issue in issues:
                        st.markdown(f"- {issue}")
            except Exception as e:
                st.error(f"Error validating expression: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown("This tool implements the numerical syntax framework from Reynolds' work on English numeratives.")
    st.markdown("Created with Streamlit and NLTK.")


if __name__ == "__main__":
    main()
