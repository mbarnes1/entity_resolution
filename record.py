# This is the main utilities file for Traffic Jam
import datetime


# A single record, i.e. ad or merge of ads
class Record(object):
    def __init__(self):
        self.ads = set()  # a set of all this person's ads

        self.posters = set()
        self.sites = set()
        self.dates = set()  # post time, using datetime object
        self.states = set()
        self.cities = set()
        self.perspective_1st = 0  # cannot satisfy ICAR
        self.perspective_3rd = 0  # cannot satisfy ICAR
        self.names = set()
        self.ages = set()
        self.costs = set()
        self.heights = set()
        self.weights = set()
        self.cups = set()
        self.chests = set()
        self.waists = set()
        self.hips = set()
        self.ethnicities = set()
        self.skincolors = set()
        self.eyecolors = set()
        self.haircolors = set()
        self.phones = set()
        self.emails = set()
        self.urls = set()
        self.medias = set()
        self.images = set()

        #self.restrictiontype = UNUSED
        #self.restrictionage = UNUSED
        #self.restrictionethnicity = UNUSED

    def merge(self, r2):
        self.phones.update(r2.phones)
        self.posters.update(r2.posters)
        self.ads.update(r2.ads)
        self.sites.update(r2.sites)
        self.dates.update(r2.dates)
        self.states.update(r2.states)
        self.cities.update(r2.cities)
        self.names.update(r2.names)
        self.ages.update(r2.ages)
        self.costs.update(r2.costs)
        self.heights.update(r2.heights)
        self.weights.update(r2.weights)
        self.cups.update(r2.cups)
        self.chests.update(r2.chests)
        self.waists.update(r2.waists)
        self.hips.update(r2.hips)
        self.ethnicities.update(r2.ethnicities)
        self.skincolors.update(r2.skincolors)
        self.eyecolors.update(r2.eyecolors)
        self.haircolors.update(r2.haircolors)
        self.emails.update(r2.emails)
        self.urls.update(r2.urls)
        self.medias.update(r2.medias)
        self.images.update(r2.images)
        self.add_perspective_1st(r2.perspective_1st)
        self.add_perspective_3rd(r2.perspective_3rd)

    def initialize_from_annotation(self, annotation, ad_number):
        features = annotation.split(',')  # extract features
        posterids = features[0].split(';')  # STRONG
        sites = features[1].split(';')  # weak
        dates = features[2].split(';')  # weak
        states = features[3].split(';')  # weak
        cities = features[4].split(';')  # weak
        # Not using 1st and 3rd perspective. Does not add up correctly with multiple merges. Would neeed to calculate
        # by referencing original ads
        perspectives_1st = features[5].split(';')  # actual count of 1st and 3rd perspectives could catch duplicates. 1st v 3rd winner
        perspectives_3rd = features[6].split(';')  # is a weak feature
        names = features[7].split(';')  # weak
        ages = features[8].split(';')  # weak
        costs = features[9].split(';')  # weak
        heights_ft = features[10].split(';')  # weak
        heights_in = features[11].split(';')  # weak
        weights = features[12].split(';')  # weak
        cups = features[13].split(';')  # weak
        chests = features[14].split(';')  # weak
        waists = features[15].split(';')  # weak
        hips = features[16].split(';')  # weak
        ethnicities = features[17].split(';')  # weak
        skincolors = features[18].split(';')  # weak
        eyecolors = features[19].split(';')  # weak
        haircolors = features[20].split(';')  # weak
        restrictions_type = features[21].split(';')  # what is this?
        restrictions_ethnicity = features[22].split(';')  # what is this?
        restrictions_age = features[23].split(';')  # what is this?
        phones = features[24].split(';')  # STRONG. extract multiple phone numbers
        areacodes_state = features[25].split(';')  # STRONG
        areacodes_cities = features[26].split(';')  # STRONG
        emails = features[27].split(';')  # STRONG
        urls = features[28].split(';')  # STRONG?
        medias = features[29].split(';')  # STRONG?
        images = features[30].split(';')  # STRONG?

        self.add_ad(ad_number)

        for posterid in posterids:
            if (posterid != '') & (posterid != 'none'):
                self.add_poster(posterid)
        for site in sites:
            if (site != '') & (site != 'none'):
                self.add_site(site)
        for date in dates:
            if (date != '') & (date != 'none'):
                self.add_date(date)
        for state in states:
            if (state != '') & (state != 'none'):
                self.add_state(state)
        for city in cities:
            if (city != '') & (city != 'none'):
                self.add_city(city)
        for perspective_1st in perspectives_1st:
            if (perspective_1st != '') & (perspective_1st != 'none'):
                self.add_perspective_1st(int(perspective_1st))
        for perspective_3rd in perspectives_3rd:
            if (perspective_3rd != '') & (perspective_3rd != 'none'):
                self.add_perspective_3rd(int(perspective_3rd))
        for name in names:
            if (name != '') & (name != 'none'):
                self.add_name(name)
        for age in ages:
            if (age != '') & (age != 'none'):
                self.add_age(int(age))
        for cost in costs:
            if (cost != '') & (cost != 'none'):
                self.add_cost(cost)
        for height_ft, height_in in zip(heights_ft, heights_in):
            if (height_ft != '') & (height_ft != 'none'):
                if (height_in == '') | (height_in == 'none'):
                    height_in = 0
                self.add_height(int(float(height_ft))*12 + int(float(height_in)))
        for weight in weights:
            if (weight != '') & (weight != 'none'):
                self.add_weight(int(float(weight)))
        for cup in cups:
            if (cup != '') & (cup != 'none'):
                self.add_cup(cup)
        for chest in chests:
            if (chest != '') & (chest != 'none'):
                self.add_chest(int(float(chest)))
        for waist in waists:
            if (waist != '') & (waist != 'none'):
                self.add_waist(int(float(waist)))
        for hip in hips:
            if (hip != '') & (hip != 'none'):
                self.add_hip(int(float(hip)))
        for ethnicity in ethnicities:
            if (ethnicity != '') & (ethnicity != 'none'):
                self.add_ethnicity(ethnicity)
        for skincolor in skincolors:
            if (skincolor != '') & (skincolor != 'none'):
                self.add_skincolor(skincolor)
        for eyecolor in eyecolors:
            if (eyecolor != '') & (eyecolor != 'none'):
                self.add_eyecolor(eyecolor)
        for haircolor in haircolors:
            if (haircolor != '') & (haircolor != 'none'):
                self.add_haircolor(haircolor)
        for phone in phones:
            if (phone != '') & (phone != 'none'):
                self.add_phone(phone)
        for email in emails:
            if (email != '') & (email != 'none'):
                self.add_email(email)
        for url in urls:
            if (url != '') & (url != 'none'):
                self.add_url(url)
        for media in medias:
            if (media != '') & (media != 'none'):
                self.add_media(media)
        for image in images:
            if (image != '') & (image != 'none'):
                self.add_image(image)

    def display(self):
        print 'Ads: ', self.ads
        print 'Phones: ', self.phones
        print 'Names: ', self.names
        print 'Poster IDs: ', self.posters
        print 'Emails: ', self.emails
        print 'Dates: ', self.dates
        print 'Sites: ', self.sites
        print 'States: ', self.states
        print 'Cities: ', self.cities
        print 'Ages: ', self.ages
        print 'Costs: ', self.costs
        print 'Heights: ', self.heights
        print 'Weights: ', self.weights
        print 'Cup sizes: ', self.cups
        print 'Chest sizes: ', self.chests
        print 'Waist sizes: ', self.waists
        print 'Hip sizes: ', self.hips
        print 'Ethnicities: ', self.ethnicities
        print 'Skin colors: ', self.skincolors
        print 'Eye colors: ', self.eyecolors
        print 'Hair colors: ', self.haircolors
        print '1st person references: ', self.perspective_1st
        print '3rd person references: ', self.perspective_3rd
        print 'Images: ', self.images
        print 'URLS: ', self.urls
        print 'Media: ', self.medias

    def add_ad(self, ad):
        self.ads.add(ad)

    def add_phone(self, phone):
        self.phones.add(phone)

    def add_poster(self, poster):
        self.posters.add(poster)

    def add_site(self, site):
        self.sites.add(site)

    def add_date(self, date):
        try:
            mo = month(date[4:7])  # month
            d = int(date[8:10])  # day
            h = int(date[11:13])  # hour
            mi = int(date[14:16])  # min
            s = int(date[17:19])  # sec
            z = date[20:23]  # time zone. Assuming EST for now
            y = int(date[24:29])  # year
            time = datetime.datetime(y, mo, d, h, mi, s)
            self.dates.add(time)
        except:
            pass

    def add_state(self, state):
        self.states.add(state)

    def add_city(self, city):
        self.cities.add(city)

    def add_name(self, name):
        self.names.add(name)

    def add_age(self, age):
        self.ages.add(age)

    def add_cost(self, cost):
        self.costs.add(cost)

    def add_height(self, height_in_inches):
        self.heights.add(height_in_inches)

    def add_weight(self, weight):
        self.weights.add(weight)

    def add_cup(self, cup):
        self.cups.add(cup)

    def add_chest(self, chest):
        self.chests.add(chest)

    def add_waist(self, waist):
        self.waists.add(waist)

    def add_hip(self, hip):
        self.hips.add(hip)

    def add_ethnicity(self, ethnicity):
        self.ethnicities.add(ethnicity)

    def add_skincolor(self, skincolor):
        self.skincolors.add(skincolor)

    def add_eyecolor(self, eyecolor):
        self.eyecolors.add(eyecolor)

    def add_haircolor(self, haircolor):
        self.haircolors.add(haircolor)

    def add_email(self, email):
        self.emails.add(email)

    def add_url(self, url):
        self.urls.add(url)

    def add_media(self, media):
        self.medias.add(media)

    def add_image(self, image):
        self.images.add(image)

    def add_perspective_1st(self, count):
        self.perspective_1st = 0  # += count  # not using, see comment above

    def add_perspective_3rd(self, count):
        self.perspective_3rd = 0  # += count

    def __eq__(self, other):
        return self.__dict__ == other.__dict__


def month(x):
    month_dict = {
        'Jan': 1,
        'Feb': 2,
        'Mar': 3,
        'Apr': 4,
        'May': 5,
        'Jun': 6,
        'Jul': 7,
        'Aug': 8,
        'Sep': 9,
        'Oct': 10,
        'Nov': 11,
        'Dec': 12,
    }
    return month_dict[x]

