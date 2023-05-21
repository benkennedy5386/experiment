import ezgmail
import sched, time
import shutil
import email
import imaplib
import mailbox
import datetime
import shutil

print('Logged in EZgmail ok = ', ezgmail.LOGGED_IN)


sch = sched.scheduler(time.time, time.sleep)
def emaildownload(sc):
    print('Downloading files')
    y = 0  #iterator
    unreadThreads = ezgmail.unread()
    for those in unreadThreads:
        thread = unreadThreads[(0+y)]
        subject = thread.messages[0].subject
        print('subeject is :',subject)
        filenum = subject[4:9]
        print(subject[14:16])
        if subject[14:16] == '01':
            thread.messages[0].downloadAllAttachments(downloadFolder="D:/models/research/object_detection/server/roofing/"+filenum,overwrite=True)
        if subject[14:16] == '02':
            thread.messages[0].downloadAllAttachments(downloadFolder="D:/models/research/object_detection/server/siding/"+filenum,overwrite=True)
        if subject[14:16] == '03':
            thread.messages[0].downloadAllAttachments(downloadFolder="D:/models/research/object_detection/server/interior/"+filenum,overwrite=True)

        y = y + 1
        print(filenum, ' has successfully downloaded attachments ')
    y = 0

    EMAIL_ACCOUNT = "ben5386simplifyai@gmail.com"
    PASSWORD = "Bruce78!"
    print('logged in ok')

    mail = imaplib.IMAP4_SSL('imap.gmail.com')
    mail.login(EMAIL_ACCOUNT, PASSWORD)
    mail.list()
    mail.select('inbox')
    result, data = mail.uid('search', None, "UNSEEN") # (ALL/UNSEEN)
    i = len(data[0].split())
    print('accessed unseen ok')

    for x in range(i):
        latest_email_uid = data[0].split()[x]
        result, email_data = mail.uid('fetch', latest_email_uid, '(RFC822)')

        raw_email = email_data[0][1]
        email_message = email.message_from_bytes(raw_email)

        date_tuple = email.utils.parsedate_tz(email_message['Date'])
        if date_tuple:
            local_date = datetime.datetime.fromtimestamp(email.utils.mktime_tz(date_tuple))
            local_message_date = "%s" %(str(local_date.strftime("%a, %d %b %Y %H:%M:%S")))
        email_from = str(email.header.make_header(email.header.decode_header(email_message['From'])))
        email_to = str(email.header.make_header(email.header.decode_header(email_message['To'])))
        subject = str(email.header.make_header(email.header.decode_header(email_message['Subject'])))


        # Body details
        for part in email_message.walk():
            if part.get_content_type() == "text/html":
                body = part.get_payload(decode=True)
                bodys=str(body)

                #extract data
                clientnameraw = bodys.find("Client Name") + 14
                clientnamerawend = bodys.find("Client Email") - 8
                clientname = bodys[clientnameraw:clientnamerawend]
                clientemailraw = bodys.find("Client Email") + 15
                clientemailend = bodys.find("Client Phone") -8
                clientemail = bodys[clientemailraw:clientemailend]
                clientphoneraw = bodys.find("Client Phone") + 15
                clientphoneend = bodys.find("Send Estimate To") -8
                clientphone = bodys[clientphoneraw:clientphoneend]
                recipientemailraw = bodys.find("Send Estimate To")+19
                recipientemailend = bodys.find("Photo 1") - 8
                recipientemail = bodys[recipientemailraw:recipientemailend]
                latlonraw = bodys.find('Latitude,Longitude') + 21
                latlonend = bodys.find('User Address')-8
                latlon = bodys[latlonraw:latlonend]
                addressraw = bodys.find('User Address') + 15
                addressend = bodys.find('USA') - 2
                address = bodys[addressraw:addressend]
                filenum = subject[4:9]
                #export all this shit to text
                info = [filenum, clientname, clientemail, clientphone, recipientemail, latlon, address]

                with open(filenum+'.txt', 'w') as filehandle:
                    for listitem in info:
                        filehandle.write('%s\n' % listitem)
                print(filenum,' has had info extracted')

                if subject[14:16] == '01':
                    shutil.move('D:/models/research/object_detection/'+filenum+'.txt', 'D:/models/research/object_detection/server/roofing/'+filenum+'.txt')
                if subject[14:16] == '02':
                    shutil.move('D:/models/research/object_detection/'+filenum+'.txt', 'D:/models/research/object_detection/server/siding/'+filenum+'.txt')
                if subject[14:16] == '03':
                    shutil.move('D:/models/research/object_detection/'+filenum+'.txt', 'D:/models/research/object_detection/server/interior/'+filenum+'.txt')

    sch.enter(120, 1, emaildownload, (sc,))

sch.enter(120, 1, emaildownload, (sch,))
sch.run()
