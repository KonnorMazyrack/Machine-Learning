from collections import Counter
import matplotlib.pyplot as plt

#Normal Emails
with open("train_N_I.txt", "r") as f:
    train_N_I = f.read().split()
with open("train_N_II.txt", "r") as f:
    train_N_II = f.read().split()
with open("train_N_III.txt", "r") as f:
    train_N_III = f.read().split()
normalEmails = train_N_I + train_N_II + train_N_III

#Spam Emails
with open("train_S_I.txt", "r") as f:
    train_S_I = f.read().split()
with open("train_S_II.txt", "r") as f:
    train_S_II = f.read().split()
with open("train_S_III.txt", "r") as f:
    train_S_III = f.read().split()
spamEmails = train_S_I + train_S_II + train_S_III

#Test Emails
with open("testEmail_I.txt", "r") as f:
    testEmail_I = f.read().split()
with open("testEmail_II.txt", "r") as f:
    testEmail_II = f.read().split()
testEmails = [testEmail_I, testEmail_II]

trainNemails = [train_N_I, train_N_II, train_N_III]
trainSemails = [train_S_I, train_S_II, train_S_III]
testnames = ["testEmail_I", "testEmail_II"]
testemails = [testEmail_I, testEmail_II]

countsN = Counter(normalEmails)
countsS = Counter(spamEmails)
key_listN = list(countsN.keys())
val_listN = list(countsN.values())
key_listS = list(countsS.keys())
val_listS = list(countsS.values())

#plot the frequency of words for Normal and Spam emails.
plt.figure(figsize=(16, 8))
plt.bar(key_listN, val_listN, color='purple')
plt.xticks(rotation=45)
plt.xlabel("Words")
plt.ylabel("Frequency")
plt.title("Normal Email Word Frequencies")
plt.show()

plt.figure(figsize=(16, 8))
plt.bar(key_listS, val_listS, color='purple')
plt.xticks(rotation=45)
plt.xlabel("Words")
plt.ylabel("Frequency")
plt.title("Spam Email Word Frequencies")
plt.show()

Nprob = []
Nwords = []
Sprob = []
Swords = []
for i in range(len(trainNemails)):
    #trainNemails
    counts = Counter(trainNemails[i])
    key_list = list(counts.keys())
    for a in range(len(key_list)):
        Nwords.append(key_list[a])
    val_list = list(counts.values())
    totalCounts = 0
    for a in val_list:
        totalCounts += a
    for a in range(len(key_list)):
        Nprob.append((val_list[a]/totalCounts))
    #trainSemails
    counts = Counter(trainSemails[i])
    key_list = list(counts.keys())
    for a in range(len(key_list)):
        Swords.append(key_list[a])
    val_list = list(counts.values())
    totalCounts = 0
    for a in val_list:
        totalCounts += a
    for a in range(len(key_list)):
        Sprob.append((val_list[a]/totalCounts))

#Classify the following two emails: testEmail_I.txt, testEmail_II.txt as to whether they are Normal or Spam
for i in range(len(testnames)):
    counts = Counter(testemails[i])
    key_list = list(counts.keys())
    val_list = list(counts.values())
    print(f'{testnames[i]}(keys): {key_list}')
    print(f'{testnames[i]}(vals): {val_list}')

    testNprobability = 0.73
    testSprobability = 0.27
    for word in key_list:
        if word in Nwords:
            index = Nwords.index(word)
            testNprobability += Nprob[index]
        if word in Swords:
            index = Swords.index(word)
            testSprobability += Sprob[index]
    if testNprobability > testSprobability:
        classification = "Normal"
    else:
        classification = "Spam"
    print(f"{testnames[i]} is classified as {classification}")