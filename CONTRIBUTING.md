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
   
7. ### Make necessary changes and commit those changes 

Add those changes to the branch you just created using the `git add` command:

```
git add .
```
Now commit those changes using the `git commit` command with a meaningful message:

```
git commit -m "Add feature: feature-name" 
```

8. ### Push changes to GitHub

Push your changes using the command `git push`:

```
git push -u origin your-branch-name
```

replacing `your-branch-name` with the name of the branch you created earlier.

9. ### Submit your changes for review

If you go to your repository on GitHub, you'll see a `Compare & pull request` button. Click on that button.

<img style="float: right;" src="https://github.com/RakshitKhajuria/test/assets/71117423/86cb56af-1dad-467e-afc1-f4594615783b" alt="create a pull request" />

10. ### Create a pull request (PR)

Create a pull request (PR) from your forked repository to the main LangTest repository.
- Clearly describe the changes you've made and the issue it addresses in the PR description.
- If the PR is related to an existing issue, mention it using the syntax "Fixes #issue_number."
> For pull request template click [here](https://github.com/JohnSnowLabs/langtest/blob/main/PULL_REQUEST_TEMPLATE.md)

<img style="float: right;" src="https://github.com/RakshitKhajuria/test/assets/71117423/5c629508-53a7-4444-b036-15f694df675c" alt="submit pull request" />

Our team will review your pull request, provide feedback, and work with you to ensure that your contribution meets the project's guidelines and quality standards

## Proposing a New Issue

1. If you have an idea for a new feature, improvement, or bug fix, you can propose a new issue.
2. Go to the [Issue Tracker](https://github.com/JohnSnowLabs/langtest/issues) and click on "New Issue" or just create a new issue from [here](https://github.com/JohnSnowLabs/langtest/issues/new) 
3. Choose the appropriate issue lables (feature, fix, bug, etc.).
4. Provide a clear and concise title and description of the proposed issue.
5. Add labels and assignees, if applicable, to help categorize and track the issue.

We're excited to have you on board with us! Together, we can make NLP models safer, more reliable, and fairer for everyone. 

### **Happy contributing!** 💫

## Where to go from here?

Congrats! You just completed the standard fork -> clone -> edit -> pull request workflow that you'll often encounter as a contributor!

Celebrate your contribution and share it with your friends and followers by going to [LinkedIn](www.linkedin.com).

You can join our community team if you need any help or have any questions. 

- [Slack](https://www.johnsnowlabs.com/slack-redirect/) For live discussion with the LangTest community, join the `#langtest` channel
- [GitHub](https://github.com/JohnSnowLabs/langtest/tree/main) For bug reports, feature requests, and contributions
- [Discussions](https://github.com/JohnSnowLabs/langtest/discussions) To engage with other community members, share ideas, and show off how you use LangTest!
  
## Contributors

We would like to acknowledge all contributors of this open-source community project. 

<a href="https://github.com/johnsnowlabs/langtest/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=johnsnowlabs/langtest" />
</a>

## License

LangTest is released under the [Apache License 2.0](https://github.com/JohnSnowLabs/langtest/blob/main/LICENSE), which guarantees commercial use, modification, distribution, patent use, private use and sets limitations on trademark use, liability and warranty.