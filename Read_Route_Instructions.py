import xml.etree.ElementTree as ET
#This file read XML file and write them to a dictionary in python
# the dictionary conveys the route instruction for each route, style and grammar combination, and origin and destination coordinates
# Create an empty dictionary to store the data
def read_route_instructions(letter_name,style_name,grammar_name,x_origin,y_origin,x_destination,y_destination):
    data = {}

    # Parse the XML file
    tree = ET.parse('data/RI/Route_Instructions.xml')

    # Get the root element
    root = tree.getroot()

    # Loop through each 'letter' element
    for letter in root.findall('letter'):
        if letter.get('name') == letter_name:

            # Loop through each 'route' element in the letter
            for route in letter.findall('route'):
                if route.get('x_origin') == x_origin and route.get('y_origin') == y_origin and route.get('x_destination') == x_destination and route.get('y_destination') == y_destination:

                # Loop through each 'style' element in the route
                    for style in route.findall('style'):
                        if style.get('name') == style_name:

                            # Loop through each 'grammar' element in the style
                            for grammar in style.findall('grammar'):
                                if grammar.get('name') == grammar_name:

                                    # Get the route instruction sentence
                                    sentence = grammar.find('route_instruction').text

                                    # Add the data to the dictionary
                                    data.setdefault(letter_name, {}).setdefault(style_name, {}).setdefault(grammar_name, []).append({
                                        'x_origin': x_origin,
                                        'y_origin': y_origin,
                                        'x_destination': x_destination,
                                        'y_destination': y_destination,
                                        'sentence': sentence
                                    })

    # print(data[letter_name][style_name][grammar_name][0]['sentence'])
    return data[letter_name][style_name][grammar_name][0]['sentence']


if __name__ == '__main__':
    read_route_instructions('A',"Turn-based","4sector","500748.678397521","1028060.3864541833","501472.0056451614","1028856.6181451613")