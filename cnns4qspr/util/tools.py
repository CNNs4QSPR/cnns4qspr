def list_parser(list_string):
    if ',' in list_string:
        list_out = "".join(list_string[1:len(list_string)-1]).split(',')
        list_out = [int(x) for x in list_out]
        return list_out
    else:
        list_out = [int(list_string[1:len(list_string)-1])]
        return list_out
