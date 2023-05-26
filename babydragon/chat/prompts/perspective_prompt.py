import json
import os
import random
# Assume script is running in the same directory as the JSON file.
# base_dir = os.path.dirname(os.path.abspath(__file__))
# filename = os.path.join(base_dir, 'babydragon', 'chat', 'prompts', 'perspective_prompts.json')


base_dir = os.path.dirname(os.path.abspath(__file__))
filename = os.path.join(base_dir, 'perspective_prompts.json')


def list_subjects_and_perspective():
    # read from the file
    prompts = json.load(open(filename))
    subject_and_perspective = list(prompts.keys())
    subjects = set()
    perspectives = set()
    for item in subject_and_perspective:
        subject, perspective = item.split('\\')
        subjects.add(subject)
        perspectives.add(perspective)
    return subjects, perspectives

def list_subjects():
    # read from the file
    prompts = json.load(open(filename))
    subject_and_perspective = list(prompts.keys())
    subjects = set()
    for item in subject_and_perspective:
        subject, perspective = item.split('\\')
        subjects.add(subject)
    return subjects

def list_perspectives():
    # read from the file
    prompts = json.load(open(filename))
    subject_and_perspective = list(prompts.keys())
    perspectives = set()
    for item in subject_and_perspective:
        subject, perspective = item.split('\\')
        perspectives.add(perspective)
    return perspectives

def get_perspective_prompt(subject, perspective):
    # read from the file
    prompts = json.load(open(filename))
    key = subject + '\\' + perspective
    if key in prompts:
        return prompts[key]
    else:
        raise Exception('No prompt found for subject: ' + subject + ' and perspective: ' + perspective +' use list_subjects() and list_perspectives() to see available prompts')

def get_random_perspective_prompt():
    # read from the file
    prompts = json.load(open(filename))
    key = random.choice(list(prompts.keys()))
    subject = key.split('\\')[0]
    perspective = key.split('\\')[1]
    return subject, perspective, prompts[key]