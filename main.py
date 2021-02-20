import os
from flask import Flask, render_template, request, redirect, send_file, session, url_for
from flask_session import Session
from werkzeug.utils import secure_filename
from GenerateForecastReport import LoadTrainedModels, GetLoadForecastingResult
from GenerateForecastReport import upload_cloud_storage, download_cloud_storage
import app_config
import msal

# Load trained ML models
Models = LoadTrainedModels()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = '/tmp'
app.config.from_object(app_config)
Session(app)

# This section is needed for url_for("foo", _external=True) to automatically
# generate http scheme when this sample is running on localhost,
# and to generate https scheme when it is deployed behind reversed proxy.
# See also https://flask.palletsprojects.com/en/1.0.x/deploying/wsgi-standalone/#proxy-setups
from werkzeug.middleware.proxy_fix import ProxyFix
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)


@app.route('/')
def index():
    if not session.get("user"):
        return redirect(url_for("login"))
    return render_template('index.html', user=session["user"])


@app.route("/login")
def login():
    # Technically we could use empty list [] as scopes to do just sign in,
    # here we choose to also collect end user consent upfront
    session["flow"] = _build_auth_code_flow(scopes=app_config.SCOPE)  #look up how to use Default as scope!
    return render_template("login.html", auth_url=session["flow"]["auth_uri"], version=msal.__version__)


@app.route('/testing')  # Its absolute URL must match your app's redirect_uri set in AAD      app_config.REDIRECT_PATH
def authorized():
    try:
        cache = _load_cache()
        result = _build_msal_app(cache=cache).acquire_token_by_auth_code_flow(
            session.get("flow", {}), request.args)
        if "error" in result:
            return render_template("auth_error.html", result=result)
        session["user"] = result.get("id_token_claims")
        print ("I got a user!")
        _save_cache(cache)
    except ValueError:  # Usually caused by CSRF
        pass  # Simply ignore them
    return redirect(url_for("index"))
#    return render_template('index.html', user=session["user"])


@app.route("/logout")
def logout():
    session.clear()  # Wipe out user and its token cache from session
    return redirect(  # Also logout from your tenant's web session
        app_config.AUTHORITY + "/oauth2/v2.0/logout" +
        "?post_logout_redirect_uri=" + url_for("index", _external=True))


@app.route('/uploadfile', methods=['GET', 'POST'])
def upload_file():
#    if not session.get("user"):
#        return redirect(url_for("login"))
#    token = _get_token_from_cache(app_config.SCOPE)
#    if not token:
#        return redirect(url_for("login"))
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            print('no file')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            print('no filename')
            return redirect(request.url)
        else:
            filename = secure_filename(file.filename)
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(save_path)
            print("saved file successfully")
            upload_cloud_storage(save_path)
            print ("saved site counts file to the cloud!")
            try:
                output_file_name = GetLoadForecastingResult(Models, filename)
                print ("Successfully generated forecast report!", "\n")
                #send file name as parameter to downlad
                return redirect('/downloadfile/'+ output_file_name)
            except:
                os.remove(save_path)
                return redirect('/ErrorFileUploaded')
    return render_template('upload_file.html')


@app.route("/ErrorFileUploaded", methods=['GET'])
def errorfile():
#    if not session.get("user"):
#        return redirect(url_for("login"))
#    token = _get_token_from_cache(app_config.SCOPE)
#    if not token:
#        return redirect(url_for("login"))
    return render_template('error_page.html')

@app.route("/downloadfile/<filename>", methods = ['GET'])
def download_file(filename):
#    if not session.get("user"):
#        return redirect(url_for("login"))
#    token = _get_token_from_cache(app_config.SCOPE)
#    if not token:
#        return redirect(url_for("login"))
    return render_template('download.html',value=filename)


@app.route('/return-files/<filename>')
def return_files_tut(filename):
#    if not session.get("user"):
#        return redirect(url_for("login"))
#    token = _get_token_from_cache(app_config.SCOPE)
#    if not token:
#        return redirect(url_for("login"))
    download_cloud_storage(filename)
    print ("Successfully downloaded ML report from Cloud Storage!!", "\n")
    file_path = '/tmp/' + filename
    return send_file(file_path, as_attachment=True, attachment_filename='')


def _load_cache():
    cache = msal.SerializableTokenCache()
    if session.get("token_cache"):
        cache.deserialize(session["token_cache"])
    return cache

def _save_cache(cache):
    if cache.has_state_changed:
        session["token_cache"] = cache.serialize()

def _build_msal_app(cache=None, authority=None):
    return msal.ConfidentialClientApplication(
        app_config.CLIENT_ID, authority=authority or app_config.AUTHORITY,
        client_credential=app_config.CLIENT_SECRET, token_cache=cache)

def _build_auth_code_flow(authority=None, scopes=None):
    return _build_msal_app(authority=authority).initiate_auth_code_flow(
        scopes or [],
        redirect_uri=url_for("authorized", _external=True))

def _get_token_from_cache(scope=None):
    cache = _load_cache()  # This web app maintains one cache per session
    cca = _build_msal_app(cache=cache)
    accounts = cca.get_accounts()
    if accounts:  # So all account(s) belong to the current signed-in user
        result = cca.acquire_token_silent(scope, account=accounts[0])
        _save_cache(cache)
        return result

app.jinja_env.globals.update(_build_auth_code_flow=_build_auth_code_flow)  # Used in template




if __name__ == "__main__":
    app.run(debug=True)
