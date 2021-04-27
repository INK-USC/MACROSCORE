import json

max_seg = [187, 330, 23, 83, 131, 95, 218, 111, 226, 149, 259, 10, 42, 212, 277, 92, 94, 221, 119, 246, 174, 253, 398, 276, 217,
           181, 230, 262, 312, 256, 374, 263, 13, 275,
           305, 417, 339, 365, 199, 92, 76, 39, 76, 21, 26, 39, 13, 24, 80, 27, 82, 24,
86, 52, 67, 29, 79, 75, 35, 50, 70, 52, 78, 72, 38, 76, 8]

data = json.load(open('./data_processed/RPP_scienceparse_classify_data.parsed_rpp', 'r'))
print("data uploaded")


def split_paper(content):
    l_total = []
    if len(content.split()) // 50 > 0:
        n = len(content.split()) // 50
    else:
        n = 1
    for w in range(n):
        if w == 0:
            l_parcial = content.split()[:100]
            l_total.append(" ".join(l_parcial))
        else:
            l_parcial = content.split()[w * 50:w * 50 + 100]
            l_total.append(" ".join(l_parcial))

    return l_total


sections = []
for r, index in zip(data, max_seg):
    whole_splitted = []
    split_content = []
    split_section = []
    for section in r['content']:
        splitted = split_paper(section['text'])
        split_content.append(splitted)
        whole_splitted.extend(splitted)
        if 'heading' in section:
            split_section.append(section['heading'])
        else:
            split_section.append('None')

    assert sum([len(a) for a in split_content]) == len(whole_splitted)

    count = 0
    prev_count = 0
    passage = None
    passagesection = None

    for content, section in zip(split_content, split_section):
        count += len(content)
        if count > index:
            passage = content[index - prev_count]
            passagesection = section
            break
        prev_count = count

    assert passage == whole_splitted[index]

    sections.append(passagesection)


section_dict = {}
for section in sections:
    if section.lower() in section_dict:
        section_dict[section.lower()] += 1
    else:
        section_dict[section.lower()] = 1



print(section_dict)

    #print(len(split_content))

