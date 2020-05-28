(function(){
	is_mobile = false;
	function getCookie(cookieName){
		var cookieValue='';
		if(document.cookie){
			var array=document.cookie.split((escape(cookieName)+'='));
			if(array.length >= 2){
				var arraySub=array[1].split(';');
				cookieValue=unescape(arraySub[0]);
			}
		}
		return cookieValue;
	}

	function getDevice() {
		ua = navigator.userAgent;
		if(typeof(dejavu_device) == 'undefined')
			is_mobile = /iPhone|iPod|iPad|Android|Windows CE|BlackBerry|Symbian|Windows Phone|webOS|Opera Mini|Opera Mobi|POLARIS|IEMobile|lgtelecom|nokia|SonyEricsson/.test(ua);
		else
			is_mobile = (dejavu_device == 'mobile');

		return {'type': (is_mobile?'mobile':'pc'), 'userAgent': ua, 'ip': '', 'os': getOS(), 'isApp': false};
	}

	function getOS(){
		ua = navigator.userAgent;
		if(/android/i.test(ua))
			return 'android';
		else if(/ios|ipad|ipod/i.test(ua))
			return 'ios';
		else
			return 'etc';
	}

	function getHost() {
		obj = document.createElement('a');
		href = (dejavu_referrer == undefined || dejavu_referrer == '') ? dejavu_referrer : document.referrer;
		obj.href = href;

		return obj.host;
	}

	function makeIframe(data, type)
	{
		ifrm = document.createElement('iframe');
		ifrm.width = is_mobile?'100%':dejavu_width;
		ifrm.height = dejavu_height;
		ifrm.frameBorder = 0;
		ifrm.marginWidth = 0;
		ifrm.marginHeight = 0;
		if(type == 'url')
			ifrm.src = data;
		else
			ifrm.srcdoc = data;

		$('body').append(ifrm);
	}

	data = {'inventory': {}, 'adt': {}, 'device': {}, 'site': {}, 'bid': {}, 'payload': {}};

	host = getHost();
	data['inventory'] = {'width': dejavu_width, 'height': dejavu_height};
	data['adt'] = {'id': getCookie('PCID'), 'type': 'pcid'};
	data['device'] = getDevice();
	data['site'] = {'id': dejavu_pub_id, 'name': host, 'domain': host, 'publisher': dejavu_pub_id};
	data['bid'] = {'minimumCpc': 0};
	data['payload'] = {'ssp_code': 'xc', 'pub_code': dejavu_pub_code, 'banner_w': dejavu_width, 'banner_h': dejavu_height, 'nw_pub_id': dejavu_pub_id};
	console.log(data);
	$.ajax({
		url: 'https://grenade.ebaykorea.com/api/getImps',
		dataType: 'json',
		headers: {'Authorization':'Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJncmVuYWRlIiwiYXVkIjoibml0bXVzIiwiZXhwIjoxOTAwODE2ODgyMDAwLCJpYXQiOjE1ODUyODQwODIwMDAsInVzZXIiOnsiaWQiOiJuaXRtdXMiLCJ0eXBlIjoic3NwIn0sImNsaWVudF9hcHAiOnsiaWQiOiIwMzQ5YzNiNy0yYWNjLTRmNTgtODA4Mi01NDZjMmUwOWFkYmUifX0.Ry__gEvUo0HSqx3nkTuntyLZhmYIg5tmvdmI5aLqU9M'},
		contentType: "application/json",
		method: 'POST',
		data: JSON.stringify(data),
		timeout: 1500,
		success: function(data, status, xhr){
			if(xhr.status == 200) {
				makeIframe(data.adm)
			} else {
				makeIframe(dejavu_passback, 'url')
			}
		},
		error: function(status) {
			makeIframe(dejavu_passback, 'url')
		}
	})
}).call(this);
