# LegalAlienChatbot: your AI language buddy, powered by Rasa Open Source

**Techies**: Jeyana Morozenko, Violetta Shishkina, Nazlı Dolu *(team TalkingHeads)*

**Mentor**: Matthijs Rijm

![Telegram chat with LegalAlien](https://raw.githubusercontent.com/Jeyana/LegalAlienBlogpost/main/images/telegram_new.jpg)

The first three sections of this blog post explain the motivation for creating **LegalAlienChatbot** and provide some Machine Learning background. [Our Experience with Rasa Open Source](#our-experience-with-rasa-open-source) section is an informally written deep dive into our journey and the technical aspects of building a chatbot on a popular free platform.

**Important**: The Rasa version we are using is 2.7.1, the Python version is 3.8

## Abstract

With our project, we aim to build a chatbot to help people overcome foreign language anxiety, which is a big part of a language barrier. This is mainly the feeling of unease, worry, nervousness and apprehension experienced in learning or using a foreign language. We believe that before directly talking with a person, some written exercise with a human-like chatbot could ease the foreign language anxiety and increase the pace of language learning. For this purpose, we used a natural language understanding model. We created a chatbot using Rasa, an open source conversational AI platform.

Unlike Grammarly, LegalAlien does not correct user's messages, but it is pretty good at understanding bad English. It can answer questions about language learning, help you develop a study plan, or simply chitchat with you. LegalAlien is available 24/7, it is never annoyed by the mistakes you make, and you can never accidentally offend it.

## Introduction

It is quite natural for people to be afraid of starting to speak a foreign language, even if they already understand a lot. This may have a significant effect on their confidence and self-esteem causing individuals to be quieter and less willing to communicate.

Language barrier is a broad term for this kind of a roadblock and it hinders communication tremendously. There are several underlying reasons, some of them are linked to experience (or language proficiency) and others are purely psychological. The fear of not being understood, the fear of misunderstanding the interlocutor, the “shame” of the non-native accent can be a huge barrier on the way to speaking a foreign language and enjoying it. If on top of that, we have a person who is an introvert, or has low self-esteem, this barrier can become an impenetrable wall that might make the person give up the idea of speaking a new language.  

One piece of advice that people with foreign language anxiety are given is to imagine how the discussion might go and think about what they want to say. Great! Prepare the required phrases and play out the scenario as if you were actually using them. Then we thought, why not do this with the help of technology?

Our goal was to build a chatbot that is fun to talk to. We thought that if the chatbot could imitate a person, users could improve themselves in personal communication until they felt confident enough to talk to real people. To achieve this, we worked with Rasa which is an open source machine learning framework for automated text and voice-based conversations. It understands user messages and constructs conversations.

So, if you are a learner of English as a foreign language, who knows basic grammar and has enough vocabulary to build simple sentences, desperately needs practice but does not feel confident to communicate with other English speakers, reach out to us and try LegalAlien! 

Our chatbot is a language buddy that is always there for you. It can help you transition from theory to practice in a fun way and on your own schedule. Learning should not be boring, otherwise it brings no long-term results. The fear of using the language can easily make any learner lose their motivation and give up. 

Remember, you are not alone on this journey of learning something new! 

## Method

In order to create a chatbot before the development of machine learning algorithms, one had to write a rule based program, where all the user messages and bot responses are hardcoded. This would require the programmer to think of every possible way that a user might say something so that it could map the correct response of the chatbot. There are two downsides of this method. One of them is the fact that it is not possible to think of every possible sentence a user might write. Another one is, even though we assume that the programmer “somehow” finds out the way of enumerating all the sentences, the user might make a typo and the algorithm won’t be able to understand this. To avoid these downsides and actually achieve what we expect our chatbot to do, we examined natural language programming (NLP) models. 

NLP is a subfield of artificial intelligence (AI) which deals with methods of programming computers to process large amounts of natural language data. However, the chatbot needs to understand what the user is saying, and it is also required to decide what to say in response. Hence, it should understand the context of data (what the user means when they are saying something). To achieve this, natural language understanding (NLU) takes the center stage. NLU is a subfield of NLP. While NLP ensures the transformation of large amounts of unstructured natural language texts into structured data, NLU focuses on deriving insights from this structured data. The use of NLU allows chatbots to comprehend the context.

To utilize an NLU model, we used Rasa which is an open source machine learning platform.

The Rasa process works as follows: 

1. The user writes a message by using a messaging channel such as Telegram, Facebook Messenger, Slack etc.

2. This message is given as an input to the NLU model which transforms the message into a machine readable format. The NLU model makes a prediction to classify the meaning of the message (intent) and returns this output to the Dialogue Management model.

3. The Dialogue Management Model decides what the chatbot should do next (action).

4. Then Rasa Open Source chooses a response to give back to the user based on the set of responses.

Rasa is able to utilize different machine learning policies to decide which action to take. The ones we implemented are Rule Policy, The Transformer Embedding Dialogue (TED) Policy and Memoization Policy. Rule Policy is used for relatively simple cases in which the model needs to classify only one message of the user. On the other hand, TED and Memoization Policies are utilized for more complex cases. The parameters of these policies can be adjusted manually through the config.yml file. For example, we experimented with the number of epochs for DIETClassifier and ResponseSelector.

## Our Experience with Rasa Open Source

Soon after the project phase started, we realized that we don't have nearly enough time to build an impressive AI chatbot from scratch. We started to look for open-source solutions we can build upon. The first attempt was with the **HuggingFace** bot, described in [this blog post](https://medium.com/huggingface/how-to-build-a-state-of-the-art-conversational-ai-with-transfer-learning-2d818ac26313). That's where we got the idea of bot personality. The app itself turned out to be unimpressive, but at least it was funny. The bot's messages are in white:

![HuggingFace](https://raw.githubusercontent.com/Jeyana/LegalAlienBlogpost/main/images/hugging_face_not_impressive.png)

After searching some more, we decided to focus on **Rasa Open Source**. It looked very promising, like the gentle waves of the Black Sea. You realize that it's full of poisonous jellyfish only when you're already neck-deep, but more on that later. 

### First success: language learning FAQ question recognized and answered correctly. Rasa Rules

After you're done with [Rasa installation](https://rasa.com/docs/rasa/installation/) (remember to choose a compatible Python version when setting up the environment!), you can run "rasa init". This creates a hello-world project with the necessary file structure. Then the official documentation is your best bet because the instructional videos and most of the stackoverflow threads about Rasa are outdated. The syntax for YAML files has changed drastically. 

This is the first conversation about language learning we managed to have with LegalAlien:

![improve English](https://raw.githubusercontent.com/Jeyana/LegalAlienBlogpost/main/images/how_improve_eng_conv.jpg)

In order to make this kind of thing possible, one has to edit three different YAML files. Great opportunity to make a typo somewhere, not get any error, and spend hours trying to figure out what went wrong!

First, **rules.yml**

This is where you write a simple script of what the bot is supposed to do. "Intent" is what the user is trying to say, "action" is what the bot does when it recognizes a particular intent.

~~~
rules:
- rule: FAQ3 Activities to improve English
  steps:
    - intent: ask_how_to_improve_english
    - action: utter_ways_to_improve_english
~~~

Then, **nlu.yml** 

You need to write some training examples for this particular user intent. What are some possible ways in which the users might express their intention? Write the ones you think are the most probable. Five examples are already a good start, because Rasa NLU models are pretrained (that is, we're benefiting from *transfer learning*).

~~~
nlu:
- intent: ask_how_to_improve_english
  examples: |
    - What daily activities can improve my language level?
    - What can I do to improve my language level?
    - Tips on language level improvement?
    - Can you give me a piece of advice on how to improve my language level?
    - How do I improve my English?
    - How to learn to speak English better?
~~~

Finally, you need to do two things in **domain.yml**.
First, declare the new intent (as you declare variables in programming languages):

~~~
intents:
- ask_how_to_improve_english
~~~

Then, write the chatbot's answer. Make sure it doesn't hurt the user's feelings. Seriously. Oh, by the way, all *response* names have to start with "utter_", otherwise it won't work.

~~~
responses:
  utter_ways_to_improve_english:
  - text: You can listen to the songs you like and learn them; watch movies with subtitles; take MOOCs you find interesting; read blogs, articles and books; write a diary in English; find new friends among native speakers and practice with them. What other fun ways can you think of?
~~~

That's it for one question (== intent) and answer.

You can make your life a bit easier by [grouping the intents](https://rasa.com/docs/rasa/chitchat-faqs), then there will be no need to edit rules.yml for every intent.

### How about something that looks like a real conversation? Rasa Stories

![conversation](https://raw.githubusercontent.com/Jeyana/LegalAlienBlogpost/main/images/your-input.png)

To implement this one, more work is required. You need to think about how the conversation might go and encode several options in **stories.yml** (forget about rules.yml for this one, rules are only for "one user message -> one bot answer" scenarios).

Here the conversation always starts with the user asking how much they should practice. In all of these stories, the chatbot starts with suggesting a study plan and asking if it is realistic for the user (using *action: utter_is_one_hour_five_days_realistic*). The user's responses can be different, and these stories tried to capture some common patterns. 

In the first story, the user chooses to go with what the chatbot suggests. In the second story, the suggested study plan is too much for the user, so the bot asks for an alternative plan. In the third story, the user says something irrelevant, and the chatbot is not insisting on answering the practice plan question.

~~~
stories:

- story: how_much_practice_geek_path
  steps:
    - intent: ask_how_much_practice
    - action: utter_is_one_hour_five_days_realistic
    - intent: affirm
    - action: utter_awesome_have_fun

- story: how_much_practice_amateur_path
  steps:
    - intent: ask_how_much_practice
    - action: utter_is_one_hour_five_days_realistic
    - intent: deny
    - action: utter_what_is_realistic_plan
    - intent: describe_study_plan
    - action: utter_good_plan

- story: how_much_practice_blah_path
  steps:
    - intent: ask_how_much_practice
    - action: utter_is_one_hour_five_days_realistic
    - intent: out_of_scope
    - action: utter_ok_ask_me_anything
~~~

Now we need to provide training examples for all the intents we need. That's again **nlu.yml**, like in the previous example.

These intents are specific to the conversation about the study plan:

~~~
nlu:
#FAQ2
- intent: ask_how_much_practice
  examples: |
    - how much should i practice?
    - for how long should i study?
    - how many hours should i study?
    - how often should i practice?
    - is five minutes of practice enough?
- intent: describe_study_plan
  examples: |
    - two days per week, 30 minutes per day
    - once a week, fifteen minutes
    - three days, one hour
    - two days, 2 hours
    - I guess 2 days, half an hour
    - I think 3 days, twenty mins
~~~

And these intents are also used in other contexts:

~~~
# GENERAL
- intent: affirm
  examples: |
    - yes
    - y
    - indeed
    - of course
    - that sounds good
    - correct
    - sure
    - ok
    
- intent: deny
  examples: |
    - no
    - n
    - never
    - I don't think so
    - don't like that
    - no way
    - not really
    - I didn't
    - nope
    
- intent: out_of_scope
  examples: |
    - that's not what I want to do
    - wait stop
    - you are no help
    - this is no help at all
    - how old are you
    - I want to order a pizza
    - this isn't working
    - I don't want to tell you that
    - none of your business
    - that's not right
~~~

Rasa doesn't care how you organize the intents and the corresponding training examples. You can try to avoid complete chaos by writing comments, but then if you run an interactive learning session (Rasa X), all your comments and formatting will be deleted, surprise! This is "a known issue".

Ok, so we have the stories and the intents. Now we need to declare the intents in **domain.yml**, like with the previous example, and write the bot's answers for all the different scenarios.

~~~
responses:
 utter_is_one_hour_five_days_realistic:
    - text: Is it realistic for you to practice five days a week, for one hour each day? Everything counts, including listening to songs and watching movies!
  utter_awesome_have_fun:
    - text: Awesome! Have fun learning English!
  utter_what_is_realistic_plan:
    - text: How many days per week can you practice? How many minutes per day? I'm curious what's your plan.
  utter_good_plan:
    - text: Nice plan! Be patient, stick to what you've decided, and you'll reach fluency!
  utter_ok_ask_me_anything:
    - text: Ok, looks like you don't want to answer. You can still ask me anything :)
~~~

Yes, it is that much work for a tiny conversation. Maybe it's easier to hire a human?

### Call me by my name. Rasa Entities and Slots

Theoretically, you can get Rasa to do virtually anything by writing Custom Actions in Python (or other programming language), but proper documentation for Custom Actions is nowhere to be found as of July 21, 2021.

We tried to guess which syntax Rasa needs from unhelpful error messages, but failed miserably. The difficult thing is not to write in Python, but to connect the ML part with the Python part in Rasa. And, by the way, even if you manage to do it, you need to run Python on a second server (Action Server), apart from all the ML stuff.

Without Custom Actions, it's impossible to save user messages, or the data extracted from them, anywhere except Rasa Slots, and only users themselves have access to those.

Anyway, it feels good when someone calls you by your name, and that's doable with Rasa Slots & Entities.

![call me Bob](https://raw.githubusercontent.com/Jeyana/LegalAlienBlogpost/main/images/hi_Bob.jpg)

Let's start with the intent in **nlu.yml**

This intent (*say_name*) is in the *chitchat* [group](https://rasa.com/docs/rasa/chitchat-faqs). After we recognize an Entity in the user message, it automatically fills the Slot with the same name (if the Slot settings are correct). We "teach" Rasa to recognize Entities in user messages like this:

~~~
nlu:
- intent: chitchat/say_name
  examples: |
    - My name is [Bob](name)
    - You can call me [Lindsey](name)
    - I am [Erik](name)
    - I'm [Ursula](name)
    - Call me [Joe](name)
~~~

Now, **domain.yml** 

Here's how the Slots and entities are defined in our case:

~~~
entities:
  - name

slots:
  name:
    type: text
    influence_conversation: true
    auto_fill: true
~~~

Notice that the entity and the slot have the same name. Together with *auto_fill: true*, it makes the autofill possible.

Also in **domain.yml**, we define the bot response, as usual, and embed the slot into it:

~~~
responses:

  utter_chitchat/say_name:
    - text: Nice to meet you, {name}!
~~~

You can define several alternative responses for one intent and let Rasa choose one of the responses randomly. You can check if the *name* slot is filled before uttering this "Nice to meet you", because otherwise you won't make your user happy:

![None](https://raw.githubusercontent.com/Jeyana/LegalAlienBlogpost/main/images/hi_None.jpg)

Rasa Open Source is huge, there's a lot to explore, and the purpose of this blog post is not to give a complete overview, but to share how it was for us to work with Rasa. So, for now, we're done with the examples.

### Advantages of Rasa

- Rasa models train pretty quickly even on an old notebook (several minutes for a simple bot).
- Even when the user message is grammatically incorrect or contains typos, the model is able to understand the intent of the user.
- Rasa trains NLU and Core models separately, and it saves time. If you make a change that has to do with only one of these models, you don't need to retrain the other one.
- The fact that it is an open-source project gives the programmer high flexibility. If you don't like something about Rasa, you are very welcome to improve it.
- Rasa is entirely written in Python and is already compatible with Python 3.8
- Training data is written in YAML files which are human-readable.

### Disadvantages of Rasa

- Debugging and collaboration can be a nightmare. Sometimes the errors are not caught, or the error messages are irrelevant and unhelpful, and you have to guess what actually went wrong.
- The syntax for YAML files changes from version to version with no backward compatibility.
- The documentation is incomplete. The "Rasa for Beginners" course on Udemy is outdated. A lot of figuring out must happen before you get anything to work.
- Machine Learning happens only when Rasa recognizes user messages and predicts bot actions. The actual bot messages are hardcoded, not generated (although they can contain variables extracted from user messages or generated with Python code).

## Conclusion

Rasa might be one of the best **free** conversational AI platforms out there, and we had a lot of fun creating this project together. Still, we wouldn't recommend Rasa for teams with more than one programmer, or for large and complex chatbots.

With this project, we also wanted to show that AI is an awesome tool that can help people acquire new skills. We even asked a GPT3-based AI what it thought about a bot that teaches English. Here’s the response:

“The idea is not to replace human teachers, but to provide some extra help to those who need it”. 

And we absolutely agree. 

## Chat with LegalAlien!

You can search for **@LegalAlienChatbot** on Telegram and write a message. The bot will be live from July 31, 2021.

If you run into problems or want to share your experience, send an email to **legalalienchatbot@gmail.com**

Thank you for your interest!


