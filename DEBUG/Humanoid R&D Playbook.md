
# Humanoid R&D Playbook

## Table of Contents
- [Who is this document for?](#who-is-this-document-for)
- [Why a R&D playbook?](#why-a-rd-playbook)
- [Things you need to know prior to coding](#things-you-need-to-know-prior-to-coding)
- [Notes](#notes)
- [Troubleshooting](#troubleshooting)
- [Humanoid Showcases](#ameca-showcases)
- [FAQs](#faqs)
- [Acknowledgements](#acknowledgements)
- [Citing](#citing)

## Who is this document for?
New members who are going to contribute something to **Humanoid Robotics** based on Ameca.

## Why a R&D playbook?
- Help reduce the time spent finding relevant materials, especially for new starters.
- Document lessons learned from past experiences to enhance knowledge sharing within our group.

## Things you need to know prior to coding

- Read the documentation from Engineered Arts (
[Engineerd Arts Uer Documentation](https://docs.engineeredarts.co.uk/)) thoroughly and familarize yourself with the web-based IDE first.

- Speech recognition and text-to-speech functions are provided by Google Cloud and Amazon Web Services respectively. 
- Basic text-only conversation is powered by ChatGPT.
- Visual reasoning (e.g., VQA, Face Recognition, Action Recognition) and other customized functions (e.g., Emotion Imitation) developed by ourselves are powered by models hosted on our local server (with one Nvidia 4090Ti GPU inside). 

- Data transmission between local server and Ameca is handled using ZMQ, a python library. A general transmission framework for the server has been developed and available here: [Amecabackend](https://github.com/lipzh5/AmecaBackend).

- Have a basic understanding of couroutines in Python and how to define them using **async/await**  (recommended reading: [Async IO in Python](https://realpython.com/async-io-python/)).

- Pay attention to the [script section](https://docs.engineeredarts.co.uk/user/scripts) of the documentation and learn how to send information and trigger functions beween script activities using **event**.

- Create your own folder under **Dev** (e.g., Dev/PLI) to help keep the environment organized.

- Feel free to contact @Penny if you have any questions or suggestions.

## Notes
- Conduct comprehensive testing for any new functions before important events to ensure they are stable (i.e., do not get stuck even under the worst conditions) and do not conflict with existing functions.

## Troubleshooting
- [Issues and solutions for Professional Staff Training Day, 2024](./Issues.md)

## Humanoid Showcases
- [Professional Staff Training Day, FSE, 2024](https://www.linkedin.com/posts/macquarie-university-faculty-of-science-and-engineering_macquarieuniversity-mqschoolofcomputing-airobot-ugcPost-7214482532543709184-Khy4/?utm_source=share&utm_medium=member_desktop)



## FAQs

## Acknowledgements

## Citing

## Contributing
- We'd like to hear your feedback! We anticipate making aperiodic improvements since this is a living document. 

- If you like this playbook, please **leave a star**!
