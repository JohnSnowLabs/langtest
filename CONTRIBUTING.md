# Contributing to LangTest

Thank you for your interest in contributing to LangTest! We value the contributions of our community members and are excited to have you on board. To ensure a smooth and collaborative development process, please follow these guidelines:

## Finding an Issue

1. **Explore Our Issues:**
Browse through our list of open issues on the [Issues tab](https://github.com/YourUsername/LangTest/issues). These issues cover a wide range of topics and difficulties. You might find something that aligns with your skills and interests.

2. **Filter and Search:**
Use the provided labels and filters to narrow down the list of issues. You can filter by tags like `feature`, `bug`, `fix`, etc. Also, utilize the GitHub search bar to look for keywords related to your expertise.

3. **Read and Understand:**
Once you've found an issue that captures your attention, take a moment to thoroughly read and understand it. Make sure you comprehend the problem, the expected solution, and any existing discussion around it.

4. **Clarify Doubts:**
If you have any questions or need further clarification, don't hesitate to comment on the issue. Our community is friendly and always willing to help. It's better to seek clarity before diving in.

5. **Claim the Issue:**
If you feel confident about tackling the issue, leave a comment expressing your intention to work on it. This prevents duplication of efforts and allows us to provide guidance if needed.

6. **Understand Contribution Guidelines:**
Familiarize yourself with our [Contribution Guidelines](https://github.com/JohnSnowLabs/langtest/wiki). This document outlines coding standards, pull request procedures, and other important details.

## Contribution Process

<img align="right" width="300" src="https://github.com/RakshitKhajuria/first-contributions/assets/71117423/be22d54d-5b62-4a23-b213-0268ed195021" alt="fork this repository" />

1. ### Fork this repository

Fork this repository by clicking on the fork button on the top of this page.
This will create a copy of this repository in your account.

2. ### Clone the repository

<img align="right" width="300" src="https://github.com/RakshitKhajuria/test/assets/71117423/b07c7d60-ab80-4b7f-b972-58fcd1f50741" alt="clone this repository" />

Now clone the forked repository to your machine. Go to your GitHub account, open the forked repository, click on the code button and then click the _copy to clipboard_ icon.

Open a terminal and run the following git command:

```
git clone "url you just copied"
```

where "url you just copied" (without the quotation marks) is the url to this repository (your fork of this project). See the previous steps to obtain the url.

<img align="right" width="300" src="https://github.com/RakshitKhajuria/test/assets/71117423/8eef9f61-283d-4bad-9e7f-8a526e33615e" alt="copy URL to clipboard" />

For example:

```
git clone https://github.com/username/langtest.git
```

where `username` is your GitHub username.

3. ### Create a branch

Change to the repository directory on your computer (if you are not already there):

```
cd langtest
```

Now create a branch using the `git checkout -b` command:

```
git checkout -b your-branch-name
```

4. **Set Up Environment**: Create a virtual environment for LangTest using your preferred method. For example, you can use `venv` or `conda`.

5. ### Install Poetry

Poetry is the package manager used for this project. If you don't have Poetry installed, run the following command:

   ```bash
   pip install poetry
   ```

6. ### Install Dependencies

Use Poetry to install the project dependencies:

   ```bash
   poetry install --with dev
   ```